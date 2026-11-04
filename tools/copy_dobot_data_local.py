#!/usr/bin/env python3
"""Copy Dobot post-sync data from OSS mount to local disk for faster training.

The script reads the training json list, copies each training.json together with
the cam1 image directory to a local mirror, and rewrites image paths inside the
local json files so that DobotVLADataset can read directly from local disk.

Usage:
    python tools/copy_dobot_data_local.py \
        --json_list /root/embodvla_zhouyuan/outputs/post_sync_data/all_training_jsons.txt \
        --local_root ./cache/dobot_vla_local \
        --workers 8
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def _copy_one_video(args) -> Tuple[str, int, int]:
    """Copy one video: training.json + cam1 images."""
    json_path, local_root, oss_prefix = args
    json_path = Path(json_path)

    # Determine local destination while preserving directory structure.
    try:
        rel = json_path.relative_to(oss_prefix)
    except ValueError:
        # Fallback: use the parent directory name.
        rel = Path(json_path.parent.name) / json_path.name
    local_json = Path(local_root) / rel
    local_dir = local_json.parent
    local_json.parent.mkdir(parents=True, exist_ok=True)

    # Load original json.
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return str(json_path), 0, 0

    frames = data.get("frames", [])
    if not frames:
        # Still copy the (empty) json so the list stays consistent.
        with open(local_json, "w") as f:
            json.dump(data, f)
        return str(json_path), 0, 0

    # Prepare local cam1 directory.
    local_cam1_dir = local_dir / "cam1"
    local_cam1_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for frame in frames:
        cam1_src = frame.get("cam1")
        if not cam1_src:
            continue
        src_path = Path(cam1_src)
        dst_path = local_cam1_dir / src_path.name
        if dst_path.exists():
            skipped += 1
        else:
            try:
                shutil.copyfile(src_path, dst_path)
                copied += 1
            except Exception:
                continue
        # Rewrite path in the local json copy.
        frame["cam1"] = str(dst_path.resolve())

    # Save rewritten json.
    with open(local_json, "w") as f:
        json.dump(data, f)

    return str(json_path), copied, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_list",
        type=str,
        default="/root/embodvla_zhouyuan/outputs/post_sync_data/all_training_jsons.txt",
        help="Path to the file listing all training.json paths.",
    )
    parser.add_argument(
        "--local_root",
        type=str,
        default="./cache/dobot_vla_local",
        help="Local directory to mirror the data into.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel copy workers.",
    )
    parser.add_argument(
        "--oss_prefix",
        type=str,
        default="/mnt/oss",
        help="Prefix of the OSS mount that will be replaced by local_root in the rewritten jsons.",
    )
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    with open(args.json_list) as f:
        json_paths = [line.strip() for line in f if line.strip()]

    print(f"[copy] {len(json_paths)} videos to copy -> {local_root}")
    print(f"[copy] Using {args.workers} workers")

    tasks = [(p, str(local_root), args.oss_prefix) for p in json_paths]

    total_copied = 0
    total_skipped = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_copy_one_video, t): t for t in tasks}
        for future in as_completed(futures):
            src, copied, skipped = future.result()
            total_copied += copied
            total_skipped += skipped
            completed += 1
            if completed % 10 == 0 or completed == len(json_paths):
                print(
                    f"[copy] {completed}/{len(json_paths)} done, "
                    f"copied {total_copied} images, skipped {total_skipped}"
                )

    # Write local training json list.
    local_list_path = local_root / "all_training_jsons_local.txt"
    with open(local_list_path, "w") as f:
        for p in json_paths:
            try:
                rel = Path(p).relative_to(args.oss_prefix)
            except ValueError:
                rel = Path(Path(p).parent.name) / Path(p).name
            local_json = (local_root / rel).resolve()
            f.write(str(local_json) + "\n")

    print(f"[copy] Done. Local list written to: {local_list_path}")
    print(f"[copy] Total images copied: {total_copied}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
