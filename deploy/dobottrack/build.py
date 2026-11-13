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


def commands(root, spec, qformer=None, vision=None, plugin=None, recipe="optimized"):
    if spec.get("recipe", recipe) != recipe:
        raise ValueError(
            "Export/build recipes differ; use a fresh export with the matching --recipe"
        )
    optimized = recipe == "optimized"
    if optimized and (
        qformer not in (None, "qformer_compact_fp")
        or vision not in (None, "vision_attention_int8")
    ):
        raise ValueError(
            "Optimized recipe requires compact FP16 QFormer and calibrated attention INT8 Vision"
        )
    qformer = qformer or ("qformer_compact_fp" if optimized else "qformer_fp")
    vision = vision or ("vision_attention_int8" if optimized else "vision")
    compile_plugin = optimized and plugin is None
    plugin = plugin or (
        root / "lib/libtrackvla_llm_norm_quant_plugin.so" if optimized else None
    )
    scripts = Path(__file__).resolve().parent
    commands = []
    if compile_plugin:
        commands.append(
            [
                "env",
                "OUTPUT_DIR=" + str(root / "lib"),
                "bash",
                str(scripts / "build_plugin.sh"),
            ]
        )
    for name in dict.fromkeys(
        [vision, qformer, "heads"]
        + (["mask"] if spec["mask_query"] and not optimized else [])
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
            # Dense external embeddings provide one context-only profile.
            "--builder_force_num_profiles",
            "1",
            "--workers",
            "1",
        ]
    )
    if plugin:
        llm = commands[-1]
        commands[-1] = [
            sys.executable,
            str(scripts / "build_llm_norm_quant_variant.py"),
            "--trackvla-plugin-lib",
            str(plugin),
        ] + llm[1:]
    return commands


def main():
    p = parser(__doc__)
    p.add_argument("--recipe", choices=["optimized", "baseline"], default="optimized")
    p.add_argument(
        "--qformer",
        choices=[
            "qformer_fp",
            "qformer_int8",
            "qformer_qkv_int8",
            "qformer_compact_fp",
        ],
        default=None,
    )
    p.add_argument(
        "--vision",
        choices=["vision", "vision_int8", "vision_int8_fc1", "vision_attention_int8"],
        default=None,
    )
    p.add_argument(
        "--plugin", type=Path, help="Build LLM with the custom norm+quant fusion plugin"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = args.work.resolve()
    spec = json.loads((root / "deployment.json").read_text())
    plan = commands(root, spec, args.qformer, args.vision, args.plugin, args.recipe)
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
    plugin = args.plugin or (
        root / "lib/libtrackvla_llm_norm_quant_plugin.so"
        if args.recipe == "optimized"
        else None
    )
    write_json(
        root / "engine_manifest.json",
        {
            "provenance": spec["provenance"],
            "recipe": args.recipe,
            "qformer": args.qformer
            or ("qformer_compact_fp" if args.recipe == "optimized" else "qformer_fp"),
            "vision": args.vision
            or ("vision_attention_int8" if args.recipe == "optimized" else "vision"),
            "plugin_sha256": sha256(plugin) if plugin else None,
            "sha256": {
                str(p.relative_to(root)): sha256(p)
                for p in (root / "engines").rglob("*.engine")
            },
        },
    )


if __name__ == "__main__":
    main()
