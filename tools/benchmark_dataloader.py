#!/usr/bin/env python3
"""Benchmark Dobot DataLoader speed (no model forward)."""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vla.data import DobotVLADataModule
from transformers import AutoTokenizer


def main():
    model_path = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    dm = DobotVLADataModule(
        training_json_list="/mnt/cpfs/dobot_vla_local/all_training_jsons_local.txt",
        tokenizer=tokenizer,
        input_size=448,
        num_history_frames=4,
        history_duration=1.0,
        n_future=8,
        future_dt=0.25,
        cam_channel="cam1",
        batch_size=1,
        num_workers=4,
        val_split=0.1,
        sample_interval=5,
        prompt="Predict the future trajectory of the ego vehicle.",
        num_image_tokens=256,
        max_length=2048,
        cache_dir="./cache/dobot_vla_2s",
        seed=42,
    )
    dm.setup("fit")
    print(f"Train dataset size: {len(dm.train_dataset)}")
    loader = dm.train_dataloader()
    print(f"Train loader length: {len(loader)}")

    print("Warming up first batch...")
    t0 = time.time()
    it = iter(loader)
    batch = next(it)
    print(f"First batch time: {time.time()-t0:.2f}s")

    print("Iterating 20 batches...")
    t0 = time.time()
    for i, _ in enumerate(loader):
        if i >= 19:
            break
    elapsed = time.time() - t0
    print(f"20 batches time: {elapsed:.2f}s, per batch: {elapsed/20:.2f}s")


if __name__ == "__main__":
    main()
