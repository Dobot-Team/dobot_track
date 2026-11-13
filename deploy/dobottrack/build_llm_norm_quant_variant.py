#!/usr/bin/env python3
"""Run ``trtllm-build`` with the TrackVLA LLM norm+quant plugins installed.

Pass everything after ``--trackvla-plugin-lib <path>`` straight through to trtllm-build,
for example:

    python3 build_llm_norm_quant_variant.py --trackvla-plugin-lib lib/xxx.so \
        --checkpoint_dir jetson_checkpoint --model_cls_file trackvla_qwen2_jetson.py \
        --model_cls_name TrackVLAQwen2Jetson --output_dir ... --builder_opt 5 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trackvla-plugin-lib", required=True)
    args, remaining = parser.parse_known_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trackvla_llm_norm_quant_plugins as trackvla_plugins
    from tensorrt_llm.commands.build import main as trtllm_build_main

    trackvla_plugins.install(os.path.abspath(args.trackvla_plugin_lib))
    sys.argv = ["trtllm-build", *remaining]
    return trtllm_build_main()


if __name__ == "__main__":
    sys.exit(main())
