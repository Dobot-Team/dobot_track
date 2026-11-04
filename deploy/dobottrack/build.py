"""Build engines on the target Orin with native TensorRT / TRT-LLM plugins."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
import subprocess
import sys
from deploy.dobottrack.common import parser, write_json, sha256


def commands(root, spec, qformer="qformer_fp", vision="vision"):
    scripts = Path(__file__).resolve().parent
    commands = []
    for name in dict.fromkeys(
        [vision, qformer, "heads"] + (["mask"] if spec["mask_query"] else [])
    ):
        commands.append(
            [
                sys.executable,
                str(scripts / "build_onnx.py"),
                "--onnx",
                str(root / f"export/{name}.onnx"),
                "--output",
                str(root / f"engines/{name}.engine"),
            ]
        )
    maximum = spec["max_sequence_length"]
    commands.append(
        [
            "trtllm-build",
            "--checkpoint_dir",
            str(root / "export/llm_checkpoint"),
            "--model_cls_file",
            str(scripts / "qwen_graph.py"),
            "--model_cls_name",
            "TrackVLAQwen2Jetson",
            "--output_dir",
            str(root / "engines/llm"),
            "--max_batch_size",
            "1",
            "--max_input_len",
            str(maximum),
            "--max_seq_len",
            str(maximum),
            "--max_num_tokens",
            str(maximum),
            "--remove_input_padding",
            "disable",
            "--paged_kv_cache",
            "disable",
            "--gpt_attention_plugin",
            "float16",
            "--builder_opt",
            "5",
            "--gemm_plugin",
            "disable",
            "--workers",
            "1",
        ]
    )
    return commands


def main():
    p = parser(__doc__)
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
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = args.work.resolve()
    spec = json.loads((root / "deployment.json").read_text())
    plan = commands(root, spec, args.qformer, args.vision)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    import tensorrt_llm

    if not str(tensorrt_llm.__version__).startswith("0.12."):
        raise RuntimeError(
            "qwen_graph.py targets TensorRT-LLM 0.12.x; use its compatible Orin environment"
        )
    logs = root / "build_logs"
    logs.mkdir(exist_ok=True)
    for i, command in enumerate(plan):
        print("BUILD", command, flush=True)
        with (logs / f"{i:02d}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(
        root / "engine_manifest.json",
        {
            "provenance": spec["provenance"],
            "qformer": args.qformer,
            "vision": args.vision,
            "sha256": {
                str(p.relative_to(root)): sha256(p)
                for p in (root / "engines").rglob("*.engine")
            },
        },
    )


if __name__ == "__main__":
    main()
