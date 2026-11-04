"""Replay causal RGB streams; record trajectory accuracy and end-to-end latency."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
import json
import tarfile
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from deploy.dobottrack.common import parser, StreamingVision, write_json, sha256
from deploy.dobottrack.collect import read_frame, validate_streams


def main():
    p = parser(__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument(
        "--qformer",
        choices=["qformer_fp", "qformer_int8", "qformer_qkv_int8"],
        default="qformer_fp",
    )
    p.add_argument(
        "--vision",
        choices=["vision", "vision_int8", "vision_int8_fc1"],
        default="vision",
    )
    p.add_argument(
        "--skip-mask-decoder",
        action="store_true",
        help="Keep the Mask query but skip decoding/visualization",
    )
    p.add_argument("--per-episode", type=int, default=10)
    args = p.parse_args()
    if args.per_episode < 1:
        raise ValueError("--per-episode must be positive")
    root = args.work.resolve()
    cfg = yaml.safe_load((root / "training_config.yaml").read_text())
    streams = json.loads((root / "streams.json").read_text())["streams"]
    validate_streams(streams)
    engines = json.loads((root / "engine_manifest.json").read_text())
    if engines["provenance"] != json.loads((root / "provenance.json").read_text()):
        raise ValueError("Engine provenance differs from export/calibration")
    for path, digest in engines["sha256"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Engine hash mismatch: {path}")
    from deploy.dobottrack.runtime_trt import Policy
    from scripts.eval_trackvla_habitat import TrackVLAHabitatAgent

    torch.set_num_threads(4)
    policy = Policy(
        root, args.base, cfg, args.qformer, args.vision, not args.skip_mask_decoder
    ).eval()
    agent = TrackVLAHabitatAgent(
        policy,
        "internvl3",
        cfg["data"],
        torch.device("cuda"),
        torch.bfloat16,
        0.1,
        0,
        True,
        "Track the person in front of you.",
    )
    cache = StreamingVision(policy, agent)
    warmup = (agent.history_size - 1) * agent.history_stride
    rows = []
    for stream in streams:
        agent.reset()
        cache.clear()
        records = stream["records"]
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
                torch.cuda.synchronize()
                start = time.perf_counter()
                agent.observe(rgb)
                _, prediction = agent.predict(record["instruction"])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if not np.isfinite(prediction).all():
                    raise RuntimeError("Nonfinite trajectory")
                if i < warmup:
                    continue
                target = np.asarray(record["trajectory"][:8], np.float32)
                error = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=-1)
                rows.append(
                    {
                        "episode": record["frames_file"],
                        "current": record["current"],
                        "split": stream["split"],
                        "ade": float(error.mean()),
                        "fde": float(error[-1]),
                        "latency_ms": elapsed,
                        "trajectory": prediction.tolist(),
                        "box": agent.last_grounding_bbox,
                        "visibility": agent.last_grounding_visibility,
                    }
                )
    valid = [r for r in rows if r["split"] == "validation"]
    if not valid:
        raise ValueError("No validation predictions")
    latencies = [r["latency_ms"] for r in valid]
    result = {
        "provenance": engines["provenance"],
        "qformer": args.qformer,
        "vision": args.vision,
        "mask_decoded": not args.skip_mask_decoder,
        "gt_boxes_used": False,
        "validation_samples": len(valid),
        "validation_ade": float(np.mean([r["ade"] for r in valid])),
        "validation_fde": float(np.mean([r["fde"] for r in valid])),
        "policy_ms": {
            "p50": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "mean": float(np.mean(latencies)),
        },
        "records": rows,
        "scope": "RGB preprocessing + cached single-frame vision + Q-identity + W8A8 LLM + task heads + CPU postprocessing; excludes archive decoding, loading and simulator/robot. ADE/FDE: held-out episodes only.",
    }
    name = f"replay_{args.vision}_{args.qformer}" + (
        "_no_mask" if args.skip_mask_decoder else ""
    )
    write_json(root / f"{name}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
