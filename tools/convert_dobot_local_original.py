#!/usr/bin/env python3
"""Convert Dobot local following data into TrackVLA-compatible JSONL format.

Only processes segments that have a non-empty scene_desc prompt (i.e. F-prefixed
segments by convention). For each aligned (frame, label) pair we build a sample
with 31 history frames, a current frame, and 8 future waypoints in the robot's
current coordinate frame. Waypoints are spaced by ``waypoint_step`` labels
(default 3) to cover ~2.4s of future trajectory at the 10Hz label rate.

Usage:
    python convert_dobot_local.py \
        --json_list /mnt/cpfs/dobot_vla_local/all_training_jsons_local.txt \
        --dst_root data_local \
        --history 31 --n_waypoints 8 --waypoint_step 3 --cam cam1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def world_to_robot(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """Convert world-frame displacement to robot-frame displacement."""
    rx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ry = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return rx, ry


def match_frames_to_labels(
    frames: List[Dict[str, Any]], labels: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Return list of (frame_index, label) using the provided frame_id mapping.

    The input JSON is assumed to have already performed time synchronization:
    each label's `frame_id` field directly indexes the corresponding frame.
    """
    matched = []
    for lbl in labels:
        idx = lbl.get("frame_id")
        if idx is None or idx < 0 or idx >= len(frames):
            continue
        matched.append((idx, lbl))
    return matched


def build_trajectory(
    labels: List[Dict[str, Any]], start_idx: int, n_waypoints: int, step: int = 1
) -> List[List[float]]:
    """Build relative waypoints from labels[start_idx+step : start_idx+step*(n_waypoints+1)].

    The input poses are in a world frame. We rotate the displacement into the
    robot's current coordinate frame (x forward, y left) using the current yaw.

    Args:
        step: spacing between consecutive waypoints in the label list. step=1
            uses consecutive labels; step=3 picks every 3rd label, extending
            the temporal horizon while keeping n_waypoints fixed.
    """
    x0 = labels[start_idx]["pose"]["x"]
    y0 = labels[start_idx]["pose"]["y"]
    yaw0 = labels[start_idx]["pose"]["yaw"]
    traj = []
    for k in range(1, n_waypoints + 1):
        p = labels[start_idx + k * step]["pose"]
        dx = p["x"] - x0
        dy = p["y"] - y0
        rx, ry = world_to_robot(dx, dy, yaw0)
        ryaw = normalize_angle(p["yaw"] - yaw0)
        traj.append([rx, ry, ryaw])
    return traj


def process_segment(
    json_path: str,
    dst_root: Path,
    history: int,
    n_waypoints: int,
    waypoint_step: int,
    cam: str,
    val_ratio: float,
    rng: Any,
    translation_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, int]]:
    """Convert one training.json into per-sample JSONL shards."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[skip] cannot load {json_path}: {exc}")
        return None

    bag_meta = data.get("bag_meta", {})
    instruction = bag_meta.get("scene_desc", "").strip()
    if not instruction:
        return None  # skip empty-prompt segments

    # Translate Chinese prompt to English if a translation map is provided
    if translation_map is not None and instruction in translation_map:
        instruction = translation_map[instruction]

    frames = data.get("frames", [])
    labels = data.get("labels", [])
    needed_labels = n_waypoints * waypoint_step + 1
    if len(frames) < history + 1 or len(labels) < needed_labels:
        return None

    frame_ts = np.array([f["timestamp"] * 1e-9 for f in frames], dtype=np.float64)
    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)

    # Map each label timestamp to the nearest frame that has the requested camera.
    # The JSON has already performed time sync, but ``frame_id`` is a per-label
    # index, not necessarily the temporally closest frame. Using nearest-frame
    # alignment makes the current image and future trajectory consistent.
    def _nearest_frame_idx(target_t: float) -> int:
        return int(np.argmin(np.abs(frame_ts - target_t)))

    label_to_frame = []
    for lbl in labels:
        fi = _nearest_frame_idx(lbl["timestamp"])
        if cam in frames[fi] and os.path.isfile(frames[fi][cam]):
            label_to_frame.append(fi)
        else:
            label_to_frame.append(None)

    # We need labels that have a valid current frame and enough future labels.
    max_start = len(labels) - n_waypoints * waypoint_step
    samples = []
    for i in range(0, max_start):
        curr_fi = label_to_frame[i]
        if curr_fi is None:
            continue

        # History: ``history`` frames immediately preceding the current frame.
        hist_fis = [
            j for j in range(curr_fi - history, curr_fi)
            if j >= 0 and cam in frames[j] and os.path.isfile(frames[j][cam])
        ]

        # Future waypoints use spaced labels to extend temporal horizon.
        try:
            waypoints = build_trajectory(labels, i, n_waypoints, waypoint_step)
        except Exception:
            continue

        curr_lbl = labels[i]
        yaw_curr = curr_lbl["pose"]["yaw"]
        vx, vy = world_to_robot(
            float(curr_lbl["velocity"]["vx"]),
            float(curr_lbl["velocity"]["vy"]),
            yaw_curr,
        )
        actions = [vx, vy, float(curr_lbl["velocity"]["omega"])]

        sample = {
            "instruction": instruction,
            "current": frames[curr_fi][cam],
            "images": [frames[idx][cam] for idx in hist_fis],
            "waypoints": waypoints,
            "trajectory": waypoints,
            "actions": actions,
            "frames_file": os.path.dirname(json_path),
        }
        samples.append(sample)

    if not samples:
        return None

    # Episode-level train/val split
    is_val = rng.random() < val_ratio
    split = "jsonl_val" if is_val else "jsonl_train"
    out_dir = dst_root / split / "dobot_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_name = Path(json_path).parent.relative_to(Path(json_path).parts[0])
    out_path = out_dir / f"{rel_name}.jsonl".replace("/", "_")

    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, separators=(",", ":"), ensure_ascii=False) + "\n")

    return {
        "split": split,
        "samples": len(samples),
        "episode": json_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_list",
        type=str,
        default="/mnt/cpfs/dobot_vla_local/all_training_jsons_local.txt",
    )
    ap.add_argument("--dst_root", type=str, default="data_local")
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--waypoint_step", type=int, default=3,
                    help="Spacing between waypoints in the label list (10Hz). "
                         "step=3 with n_waypoints=8 gives ~2.4s future horizon.")
    ap.add_argument("--cam", type=str, default="cam1")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--prompt_translation",
        type=str,
        default="data_local/prompt_translation.json",
        help="JSON dict mapping Chinese prompts to English translations",
    )
    args = ap.parse_args()

    import random

    rng = random.Random(args.seed)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    translation_map: Optional[Dict[str, str]] = None
    trans_path = Path(args.prompt_translation)
    if trans_path.exists():
        with open(trans_path) as f:
            translation_map = json.load(f)
        print(f"Loaded prompt translations: {len(translation_map)} entries")

    with open(args.json_list) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    # Process only segments with prompts (F-prefixed by convention)
    results = []
    for p in all_paths:
        seg_name = Path(p).parts[-3]  # e.g. F-01-016-004-...
        if seg_name.startswith("N-"):
            continue
        res = process_segment(
            p, dst_root, args.history, args.n_waypoints, args.waypoint_step, args.cam,
            args.val_ratio, rng, translation_map,
        )
        if res:
            results.append(res)

    train = sum(1 for r in results if r["split"] == "jsonl_train")
    val = sum(1 for r in results if r["split"] == "jsonl_val")
    total_samples = sum(r["samples"] for r in results)
    print(
        f"[done] episodes: train={train}, val={val}, total_samples={total_samples}"
    )


if __name__ == "__main__":
    main()
