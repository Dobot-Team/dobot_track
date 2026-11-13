"""Run the complete Box-before-Action optimized deployment recipe."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def commands(args):
    def call(script, *values):
        return [sys.executable, str(HERE / script), *map(str, values)]

    model = [
        "--work",
        args.work,
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--base",
        args.base,
        "--sam",
        args.sam,
    ]
    if args.jetson:
        model.append("--jetson")
    plan = [
        (
            "export",
            call(
                "export.py",
                *model,
                "--instructions",
                args.instructions,
                "--recipe",
                args.recipe,
            ),
        ),
        (
            "collect",
            call(
                "collect.py", *model, "--data", args.data, "--manifest", args.manifest
            ),
        ),
        (
            "llm_quantize",
            call(
                "quantize_llm.py",
                "--model-dir",
                args.work / "export/qwen_hf",
                "--calibration",
                args.work / "llm_calibration.npz",
                "--samples",
                args.samples,
                "--smoothing-strength",
                "0.8",
                "--output",
                args.work / "export/llm_checkpoint",
            ),
        ),
    ]
    # All vision kernels remain TensorRT-native; no custom ViT plugin is used.
    if args.recipe == "optimized":
        plan += [
            ("prepare_vision", call("prepare_vision.py", *model, "--data", args.data)),
            ("vision_entropy", call("quantize_vision_entropy.py", "--work", args.work)),
        ]
    if args.vision == "vision_int8" and args.recipe == "baseline":
        plan.append(
            (
                "vision_quantize",
                call(
                    "quantize_vision.py",
                    *model,
                    "--data",
                    args.data,
                    "--fc2-prefix",
                    12,
                ),
            )
        )
    build = call("build.py", "--work", args.work, "--recipe", args.recipe)
    if args.vision:
        build += ["--vision", args.vision]
    if args.fuse_llm_norm_quant and args.recipe == "baseline":
        library = args.work / "lib/libtrackvla_llm_norm_quant_plugin.so"
        plan.append(
            (
                "llm_plugin",
                [
                    "env",
                    "OUTPUT_DIR=" + str(args.work / "lib"),
                    "bash",
                    str(HERE / "build_plugin.sh"),
                ],
            )
        )
        build += ["--plugin", str(library)]
    plan.append(("build", build))
    plan.append(
        (
            "replay",
            call(
                "replay.py",
                "--work",
                args.work,
                "--base",
                args.base,
                "--data",
                args.data,
                "--recipe",
                args.recipe,
            ),
        )
    )
    return plan


def outputs(root, stage):
    patterns = {
        "export": [
            "export/vision.onnx",
            "export/qformer_fp.onnx",
            "export/qformer_compact_fp.onnx",
            "export/heads.onnx",
            "export/qwen_hf/*",
            "templates.*",
            "deployment.json",
            "provenance.json",
            "training_config.yaml",
        ],
        "collect": [
            "samples.json",
            "streams.json",
            "collection.json",
            "llm_calibration.npz",
            "captured/*.npz",
        ],
        "prepare_vision": [
            "export/vision_sq.onnx",
            "vision_calibration.json",
            "vision_entropy64.npz",
        ],
        "vision_entropy": [
            "export/vision_mlp_int8.onnx",
            "export/vision_attention_int8.onnx",
        ],
        "llm_quantize": ["export/llm_checkpoint/*"],
        "vision_quantize": ["export/vision_int8.onnx"],
        "llm_plugin": ["lib/*.so"],
        "build": ["engines/**/*.engine", "engine_manifest.json"],
        "replay": ["replay_*.json"],
    }
    return sorted(
        {p for pattern in patterns[stage] for p in root.glob(pattern) if p.is_file()}
    )


def valid_outputs(receipt):
    return bool(receipt) and all(
        Path(p).is_file() and digest(p) == h for p, h in receipt.items()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "work",
        "checkpoint",
        "config",
        "base",
        "sam",
        "data",
        "manifest",
        "instructions",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--jetson", action="store_true")
    p.add_argument("--recipe", choices=["optimized", "baseline"], default="optimized")
    p.add_argument(
        "--vision",
        choices=["vision", "vision_int8", "vision_attention_int8"],
        default=None,
    )
    p.add_argument("--fuse-llm-norm-quant", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for name in [
        "work",
        "checkpoint",
        "config",
        "base",
        "sam",
        "data",
        "manifest",
        "instructions",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    if args.samples < 1:
        p.error("--samples must be positive")
    plan = commands(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repository = HERE.parents[1]
        source = {
            str(p.relative_to(repository)): digest(p)
            for folder in [HERE, repository / "vla", repository / "scripts"]
            for p in folder.rglob("*")
            if p.suffix in [".py", ".sh", ".cu"]
        }
        inputs = {
            str(p): digest(p)
            for p in [
                args.checkpoint,
                args.config,
                args.sam,
                args.manifest,
                args.instructions,
            ]
        }
        # Archive contents and local base model also determine calibration/output identity.
        streams = json.loads(args.manifest.read_text())["streams"]
        archives = {
            record["frames_file"] for stream in streams for record in stream["records"]
        }
        for path in archives:
            rel = Path(path)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Archive path must be relative to --data")
            inputs[str(args.data / rel)] = digest(args.data / rel)
        for path in args.base.rglob("*"):
            if path.is_file():
                inputs[str(path)] = digest(path)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"source": source, "inputs": inputs, "commands": plan}, sort_keys=True
            ).encode()
        ).hexdigest()
        statepath = args.work / "workflow_state.json"
        if statepath.exists():
            state = json.loads(statepath.read_text())
            if state["fingerprint"] != fingerprint:
                raise ValueError(
                    "Inputs/recipe/source changed; use a fresh work directory"
                )
        else:
            if any(x.name != "workflow.lock" for x in args.work.iterdir()):
                raise ValueError("Use a fresh work directory for the first run")
            state = {
                "fingerprint": fingerprint,
                "source": source,
                "inputs": inputs,
                "stages": {},
            }
        (args.work / "workflow_logs").mkdir(exist_ok=True)
        for name, cmd in plan:
            prior = state["stages"].get(name, {})
            if prior.get("status") == "complete" and valid_outputs(
                prior.get("outputs", {})
            ):
                print("VERIFIED_SKIP", name, flush=True)
                continue
            if name == "llm_quantize":
                available = json.loads((args.work / "collection.json").read_text())[
                    "calibration_samples"
                ]
                if args.samples > available:
                    raise ValueError(
                        f"Only {available} calibration samples; lower --samples"
                    )
            env = None
            if name == "replay" and args.fuse_llm_norm_quant:
                import os

                env = dict(
                    os.environ,
                    DOBOTTRACK_LLM_PLUGIN=str(
                        args.work / "lib/libtrackvla_llm_norm_quant_plugin.so"
                    ),
                )
            print("RUN", name, flush=True)
            state["active"] = name
            write(statepath, state)
            try:
                with (args.work / "workflow_logs" / f"{name}.log").open("w") as log:
                    subprocess.run(
                        cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env
                    )
                generated = outputs(args.work, name)
                if not generated:
                    raise RuntimeError("No output receipt for " + name)
                state["stages"][name] = {
                    "status": "complete",
                    "outputs": {str(p): digest(p) for p in generated},
                }
            except BaseException as e:
                state["stages"][name] = {"status": "failed", "error": str(e)}
                write(statepath, state)
                raise
            write(statepath, state)
        state["active"] = None
        write(statepath, state)


if __name__ == "__main__":
    main()
