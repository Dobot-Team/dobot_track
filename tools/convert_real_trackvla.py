#!/usr/bin/env python3
"""Convert post_sync_res real robot bags to TrackVLA training archives.

The source bags are never modified.  The converter writes RGB and mask ZIP
archives plus a lazy-loadable dataset.json under a separate output root.
Trajectory targets are future relative poses in the current ego frame.
Distance and facing are deliberately left as null because the real labels do
not contain those annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import zipfile
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--future", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--training-file", type=Path, default=None,
                   help="Convert one training.json (useful for a smoke test).")
    p.add_argument("--bbox-vqa-jsonl", type=Path, default=None,
                   help="Existing follow_bbox_vqa.jsonl labels.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose(current: Dict[str, Any], future: Dict[str, Any]) -> List[float]:
    p0 = current["pose"]
    p1 = future["pose"]
    dx = float(p1["x"]) - float(p0["x"])
    dy = float(p1["y"]) - float(p0["y"])
    yaw = float(p0["yaw"])
    # World XY -> current robot ego frame: x forward, y lateral.
    return [
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
        wrap_angle(float(p1["yaw"]) - yaw),
    ]


def nearest_label_index(times: List[float], timestamp_ns: int) -> int:
    target = float(timestamp_ns) / 1e9
    pos = bisect_left(times, target)
    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    return pos if abs(times[pos] - target) < abs(times[pos - 1] - target) else pos - 1


def mask_stats(mask_path: Path) -> Tuple[List[float], int, int, int]:
    mask = np.load(mask_path, mmap_mode="r")
    foreground = np.asarray(mask) > 0
    ys, xs = np.nonzero(foreground)
    h, w = foreground.shape
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    return [
        (x0 + x1) / (2.0 * w),
        (y0 + y1) / (2.0 * h),
        bw / w,
        bh / h,
    ], int(foreground.sum()), w, h


def resolve_local(path_value: str, source_root: Path) -> Path:
    """Map /dataset/post_sync_res/pipeline_v2/... to local source_root/... ."""
    path = Path(path_value)
    marker = Path("post_sync_res") / "pipeline_v2"
    parts = path.parts
    try:
        idx = parts.index("post_sync_res")
        rel = Path(*parts[idx + 2 :])
        return source_root / rel
    except ValueError:
        return path if path.is_absolute() else source_root / path


def build_label_index(jsonl_path: Path, db_path: Path) -> sqlite3.Connection:
    """Build a disk-backed index without holding the 600+ MB JSONL in RAM."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS labels (k TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    if conn.execute("SELECT 1 FROM labels LIMIT 1").fetchone() is None:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            rows = []
            for line_no, line in enumerate(handle, 1):
                item = json.loads(line)
                image_name = Path(str(item.get("image", ""))).stem
                key = "|".join((str(item.get("date", "")), str(item.get("episode", "")), str(item.get("bag", "")), image_name))
                rows.append((key, json.dumps({
                    "bbox_cxcywh_norm": item.get("bbox_cxcywh_norm"),
                    "position": item.get("position"),
                    "scale": item.get("scale"),
                    "target_distance": item.get("target_distance"),
                    "facing": item.get("facing"),
                }, ensure_ascii=False)))
                if len(rows) >= 5000:
                    conn.executemany("INSERT OR REPLACE INTO labels(k,payload) VALUES (?,?)", rows)
                    conn.commit()
                    rows.clear()
                if line_no % 100000 == 0:
                    print(f"[labels] indexed {line_no} lines", flush=True)
            if rows:
                conn.executemany("INSERT OR REPLACE INTO labels(k,payload) VALUES (?,?)", rows)
                conn.commit()
    return conn


def label_for(conn: Optional[sqlite3.Connection], key: str) -> Dict[str, Any]:
    if conn is None:
        return {}
    row = conn.execute("SELECT payload FROM labels WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else {}


def convert_episode(
    training_path: Path,
    source_root: Path,
    output_root: Path,
    future: int,
    frame_stride: int,
    label_db: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    data = json.loads(training_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    labels = data.get("labels", [])
    if not frames or not labels:
        return []
    label_times = [float(item["timestamp"]) for item in labels]

    bag_root = training_path.parent
    date, episode, bag = training_path.parts[-4:-1]
    episode_key = "__".join(training_path.parts[-4:-1])
    rgb_zip_path = output_root / "frames" / f"{episode_key}.jpgs"
    mask_zip_path = output_root / "masks" / f"{episode_key}.masks"
    rgb_zip_path.parent.mkdir(parents=True, exist_ok=True)
    mask_zip_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    # Write archives atomically so an interrupted conversion cannot leave a
    # dataset.json pointing at a partially written episode.
    rgb_tmp = rgb_zip_path.with_suffix(rgb_zip_path.suffix + ".tmp")
    mask_tmp = mask_zip_path.with_suffix(mask_zip_path.suffix + ".tmp")
    with zipfile.ZipFile(rgb_tmp, "w", compression=zipfile.ZIP_STORED) as rgb_zip, \
        zipfile.ZipFile(mask_tmp, "w", compression=zipfile.ZIP_STORED) as mask_zip:
        frame_no = 0
        for frame_index in range(0, len(frames), frame_stride):
            frame = frames[frame_index]
            rgb_path = resolve_local(str(frame.get("undist_cam1", "")), source_root)
            if not rgb_path.is_file():
                continue
            timestamp_ns = int(frame.get("timestamp", 0))
            label_index = nearest_label_index(label_times, timestamp_ns)
            if label_index + future >= len(labels):
                continue
            stem = f"frame_{frame_no:06d}"
            frame_name = f"{stem}.jpg"
            mask_path = bag_root / "mask_undist_cam1" / (rgb_path.stem + ".npy")
            if not mask_path.is_file():
                continue
            bbox, pixel_count, width, height = mask_stats(mask_path)
            existing = label_for(label_db, "|".join((date, episode, bag, rgb_path.stem)))
            if existing.get("bbox_cxcywh_norm"):
                bbox = existing["bbox_cxcywh_norm"]
            rgb_zip.write(rgb_path, frame_name)
            # The loader decodes masks as grayscale images from the mask ZIP.
            mask_array = (np.load(mask_path) > 0).astype(np.uint8) * 255
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.fromarray(mask_array).save(buf, format="PNG")
            mask_zip.writestr(frame_name, buf.getvalue())
            trajectory = [
                relative_pose(labels[label_index], labels[label_index + step])
                for step in range(1, future + 1)
            ]
            area = bbox[2] * bbox[3]
            position = existing.get("position") or ("left" if bbox[0] < 1 / 3 else ("right" if bbox[0] >= 2 / 3 else "center"))
            scale = existing.get("scale") or ("small" if area < 0.05 else ("large" if area >= 0.20 else "medium"))
            records.append({
                "frames_file": f"frames/{episode_key}.jpgs",
                "current": frame_name,
                "mask_file": f"masks/{episode_key}.masks",
                "pixel_count": pixel_count,
                "instruction": str(data.get("bag_meta", {}).get("scene_desc", "Follow the person ahead.")),
                "trajectory": trajectory,
                "bbox_cxcywh_norm": bbox,
                "bbox_area_norm": area,
                "position": position,
                "scale": scale,
                "target_distance": existing.get("target_distance"),
                "facing": existing.get("facing"),
                "source_frame_index": frame_index,
                "source_timestamp": timestamp_ns,
            })
            frame_no += 1
    os.replace(rgb_tmp, rgb_zip_path)
    os.replace(mask_tmp, mask_zip_path)
    return records


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and args.overwrite:
        # Only the explicitly requested output root is removed.
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    label_db = None
    if args.bbox_vqa_jsonl is not None:
        index_path = output_root / "bbox_vqa_labels.sqlite"
        print(f"[labels] building index at {index_path}", flush=True)
        label_db = build_label_index(args.bbox_vqa_jsonl.resolve(), index_path)
    manifests: List[Dict[str, Any]] = []
    if args.training_file is not None:
        training_files = [args.training_file.resolve()]
    else:
        training_files = sorted(source_root.glob("*/**/training.json"))
    if args.max_episodes:
        training_files = training_files[: args.max_episodes]
    print(f"[convert] episodes={len(training_files)} source={source_root}")
    for i, training_path in enumerate(training_files, 1):
        records = convert_episode(training_path, source_root, output_root, args.future, args.frame_stride, label_db)
        manifests.extend(records)
        if i == 1 or i % 10 == 0:
            print(f"[convert] {i}/{len(training_files)} episodes, samples={len(manifests)}", flush=True)
    (output_root / "dataset.json").write_text(json.dumps(manifests, ensure_ascii=False), encoding="utf-8")
    (output_root / "conversion_summary.json").write_text(json.dumps({
        "episodes": len(training_files),
        "samples": len(manifests),
        "future": args.future,
        "frame_stride": args.frame_stride,
        "distance_supervision": False,
        "facing_supervision": False,
    }, indent=2), encoding="utf-8")
    print(f"[convert] wrote {output_root / 'dataset.json'} samples={len(manifests)}")
    if label_db is not None:
        label_db.close()


if __name__ == "__main__":
    main()
