#!/usr/bin/env python3
"""Verify that DobotVLADataset prepends/augments bag_meta.scene_desc in prompts.

Usage:
    /root/embodvla/.venv/bin/python tools/verify_scene_desc_prompt.py \
        --config configs/dobot_internvl2_5_vla.yaml \
        --max_samples 5
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml
from transformers import AutoTokenizer

from vla.data.dobot_vla import DobotVLADataset
from vla.utils.common import build_transform


class PrintingTokenizer:
    """Wraps a tokenizer and prints the raw text before tokenization."""

    def __init__(self, real_tokenizer):
        self.real_tokenizer = real_tokenizer

    @staticmethod
    def _compact(text: str, num_image_tokens: int) -> str:
        placeholder = f"<img><IMG_CONTEXT x{num_image_tokens}></img>"
        # Collapse each full image-token block to one line.
        target = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
        return text.replace(target, placeholder)

    def __call__(self, text, **kwargs):
        compact = self._compact(text, kwargs.get("num_image_tokens", 256))
        print("\n" + "=" * 80)
        print("PROMPT TEXT (compact, before tokenization):")
        print("=" * 80)
        print(compact)
        print("=" * 80 + "\n")
        return self.real_tokenizer(text, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dobot_internvl2_5_vla.yaml")
    parser.add_argument("--json", type=str, default=None, help="Single training.json to verify.")
    parser.add_argument("--max_samples", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["model_path"]
    print(f"[verify] Loading tokenizer from {model_path} ...")
    real_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.json:
        json_paths = [args.json]
    else:
        list_path = config["data"]["training_json_list"]
        with open(list_path) as f:
            all_paths = [line.strip() for line in f if line.strip()]

        # Pick the first video with scene_desc and the first without.
        with_path = None
        without_path = None
        for p in all_paths:
            if not os.path.exists(p):
                continue
            data = json.load(open(p))
            if len(data.get("frames", [])) == 0 or len(data.get("labels", [])) == 0:
                continue
            sd = data.get("bag_meta", {}).get("scene_desc", "")
            if sd and with_path is None:
                with_path = p
            elif not sd and without_path is None:
                without_path = p
            if with_path is not None and without_path is not None:
                break
        json_paths = [p for p in [with_path, without_path] if p is not None]
        print(f"[verify] Selected {len(json_paths)} videos (with/without scene_desc).")
        for p in json_paths:
            sd = json.load(open(p)).get("bag_meta", {}).get("scene_desc", "")
            print(f"  - {p}\n    scene_desc={bool(sd)} len={len(sd)}")

    if not json_paths:
        print("[verify] No valid json found, exiting.")
        return

    input_size = config["data"]["input_size"]
    transform = build_transform(input_size)

    dataset = DobotVLADataset(
        json_paths=json_paths,
        tokenizer=PrintingTokenizer(real_tokenizer),
        transform=transform,
        num_history_frames=config["data"]["num_history_frames"],
        history_duration=config["data"].get("history_duration", 1.0),
        n_future=config["data"]["n_future"],
        future_dt=config["data"].get("future_dt", 0.25),
        cam_channel=config["data"]["cam_channel"],
        sample_interval=config["data"].get("sample_interval", 5),
        prompt=config["data"]["prompt"],
        num_image_tokens=config["data"].get("num_image_tokens", 256),
        max_length=config["data"].get("max_length", 2048),
        cache_dir=None,  # avoid side effects
        num_build_workers=1,
        use_scene_desc=True,
        scene_desc_augment=True,
        scene_desc_seed=config["data"].get("seed", 42),
    )

    print(f"[verify] Dataset has {len(dataset)} samples. Printing first {args.max_samples} prompts...")
    for i in range(min(args.max_samples, len(dataset))):
        print(f"\n--- sample {i} ---")
        _ = dataset[i]


if __name__ == "__main__":
    main()
