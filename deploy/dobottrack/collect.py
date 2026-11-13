"""Collect causal predicted-box calibration streams, never ground-truth boxes."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
import json
import tarfile
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from deploy.dobottrack.common import model_parser, load, StreamingVision, write_json


def read_frame(archive, name):
    if isinstance(archive, zipfile.ZipFile):
        return archive.read(name)
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Not a frame file: {name}")
    return member.read()


def validate_streams(streams):
    """Require explicit splits and disjoint archives, before any model loading."""
    seen = {"calibration": set(), "validation": set()}
    if not isinstance(streams, list) or not streams:
        raise ValueError("Manifest needs a nonempty streams list")
    for stream in streams:
        if stream.get("split") not in seen or not stream.get("records"):
            raise ValueError(
                "Each stream needs split=calibration|validation and records"
            )
        for record in stream["records"]:
            if not isinstance(record.get("current"), str) or not isinstance(
                record.get("instruction"), str
            ):
                raise ValueError("Each record needs current and instruction strings")
            trajectory = np.asarray(record.get("trajectory"), dtype=np.float32)
            if (
                trajectory.ndim != 2
                or trajectory.shape[0] < 8
                or trajectory.shape[1] != 3
                or not np.isfinite(trajectory).all()
            ):
                raise ValueError(
                    "Each record needs at least eight finite (x,y,yaw) future points"
                )
        archives = {r["frames_file"] for r in stream["records"]}
        if len(archives) != 1:
            raise ValueError("Each stream must reference a single episode archive")
        seen[stream["split"]].update(archives)
    if not all(seen.values()) or seen["calibration"] & seen["validation"]:
        raise ValueError(
            "Use nonempty, episode-disjoint calibration and validation splits"
        )


def main():
    p = model_parser(__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument(
        "--data", required=True, type=Path, help="Root for frames_file paths"
    )
    p.add_argument("--per-episode", type=int, default=10)
    args = p.parse_args()
    streams = json.loads(args.manifest.read_text())["streams"]
    validate_streams(streams)
    if args.per_episode < 1:
        raise ValueError("--per-episode must be positive")
    r, cfg, model, agent = load(args)
    cache = StreamingVision(model, agent)
    staging, samples = {}, []
    out = r / "captured"
    out.mkdir(exist_ok=True)

    def compressor_hook(module, values, kwargs):
        staging["features"] = (
            values[0].detach().float().cpu().numpy()[0].astype(np.float16)
        )
        staging["boxes"] = kwargs["history_boxes"].detach().float().cpu().numpy()[0]
        staging["valid"] = kwargs["history_box_valid"].detach().cpu().numpy()[0]

    def llm_hook(module, values, kwargs):
        for name in ["inputs_embeds", "attention_mask", "position_ids"]:
            staging[name] = (
                kwargs[name].detach().cpu().float().numpy()[0]
                if name == "inputs_embeds"
                else kwargs[name].detach().cpu().numpy()[0]
            )
        staging["inputs_embeds"] = staging["inputs_embeds"].astype(np.float16)

    def output_hook(module, values, output):
        staging["trajectory"] = output["pred_traj"].detach().float().cpu().numpy()[0]

    model.history_compressor.register_forward_pre_hook(
        compressor_hook, with_kwargs=True
    )
    model.internvl.language_model.register_forward_pre_hook(llm_hook, with_kwargs=True)
    model.register_forward_hook(output_hook)
    warmup = (agent.history_size - 1) * agent.history_stride
    for stream in streams:
        records = stream["records"]
        if len(records) <= warmup:
            raise ValueError(
                f"Each stream needs more than {warmup} records for causal history warmup"
            )
        agent.reset()
        cache.clear()
        archive_path = args.data / records[0]["frames_file"]
        archive = (
            zipfile.ZipFile(archive_path)
            if zipfile.is_zipfile(archive_path)
            else tarfile.open(archive_path)
        )
        with archive:
            for i, record in enumerate(records[: warmup + args.per_episode]):
                rgb = np.asarray(
                    Image.open(
                        io.BytesIO(read_frame(archive, record["current"]))
                    ).convert("RGB")
                )
                agent.observe(rgb)
                cache.new_frames = 0
                agent.predict(record["instruction"])
                if cache.new_frames != 1:
                    raise RuntimeError(
                        "Streaming cache must encode exactly one new frame per observation"
                    )
                if i < warmup:
                    continue
                name = f"{len(samples):05d}.npz"
                np.savez_compressed(
                    out / name,
                    **staging,
                    target=np.asarray(record["trajectory"][:8], np.float32),
                )
                samples.append(
                    {
                        "index": len(samples),
                        "path": name,
                        "episode": record["frames_file"],
                        "current": record["current"],
                        "instruction": record["instruction"],
                        "split": stream["split"],
                    }
                )
    write_json(r / "samples.json", samples)
    write_json(r / "streams.json", {"streams": streams})
    # Variable prompt lengths are right-padded; only calibration episodes enter SQ ranges.
    selected = [s for s in samples if s["split"] == "calibration"]
    arrays = [np.load(out / s["path"]) for s in selected]
    length = max(len(x["inputs_embeds"]) for x in arrays)
    embeds = np.zeros((len(arrays), length, 896), np.float16)
    masks = np.zeros((len(arrays), length), np.int64)
    positions = np.zeros((len(arrays), length), np.int64)
    for i, x in enumerate(arrays):
        n = len(x["inputs_embeds"])
        embeds[i, :n] = x["inputs_embeds"]
        masks[i, :n] = x["attention_mask"]
        positions[i, :n] = x["position_ids"]
        x.close()
    np.savez_compressed(
        r / "llm_calibration.npz",
        inputs_embeds=embeds,
        attention_mask=masks,
        position_ids=positions,
    )
    write_json(
        r / "collection.json",
        {
            "samples": len(samples),
            "calibration_samples": len(selected),
            "gt_boxes_used": False,
            "source_frames_encoded": cache.total_source,
            "provenance": json.loads((r / "provenance.json").read_text()),
        },
    )
    print("COLLECTED", len(samples), "CALIBRATION", len(selected), flush=True)


if __name__ == "__main__":
    main()
