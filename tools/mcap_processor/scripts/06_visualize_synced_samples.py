#!/usr/bin/env python3
"""
Visualize synchronized VLA samples.

This script can:
  - Display a single sample (image + pointcloud + metadata)
  - Plot time synchronization offset distribution across all samples
  - Plot IMU data for a single sample

All outputs are saved as image files (no GUI required).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # headless backend

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcap_processor.utils import ensure_dir


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SYNC_DIR = str(PROJECT_ROOT / "output" / "synced_vla_samples")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "visualizations")


def load_index(sync_dir: Path) -> dict:
    index_path = sync_dir / "index.json"
    with open(index_path) as f:
        return json.load(f)


def load_sample(sample_dir: Path) -> dict[str, Any]:
    """Load all camera images, pointcloud and metadata for one sample."""
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)

    sync_dir = sample_dir.parent

    def _load_image(path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path.resolve()))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    cameras = {"ref": _load_image(sample_dir / "image.png")}
    for topic, info in metadata.get("cameras", {}).items():
        if info and "path" in info:
            cameras[topic] = _load_image(sync_dir / info["path"])

    pointcloud = np.load((sample_dir / "pointcloud.npy").resolve())

    return {
        "metadata": metadata,
        "cameras": cameras,
        "pointcloud": pointcloud,
    }


def visualize_single_sample(
    sample_dir: Path,
    output_path: Path,
    max_points: int = 5000,
) -> None:
    """Create a figure with all cameras, pointcloud views, and metadata."""
    sample = load_sample(sample_dir)
    meta = sample["metadata"]
    cameras = sample["cameras"]
    pc = sample["pointcloud"]

    num_extra_cameras = len(meta.get("cameras", {}))
    num_camera_axes = 1 + num_extra_cameras  # ref + extras

    fig = plt.figure(figsize=(18, 12))
    # Top row: cameras; Middle row: pointclouds; Bottom: metadata
    gs = fig.add_gridspec(3, max(num_camera_axes, 3), height_ratios=[1, 1, 0.8])

    # 1. All camera images
    ax_ref = fig.add_subplot(gs[0, 0])
    if cameras.get("ref") is not None:
        ax_ref.imshow(cameras["ref"])
    else:
        ax_ref.text(0.5, 0.5, "Ref image not found", ha="center", va="center")
    ax_ref.set_title("Reference: color")
    ax_ref.axis("off")

    for col, (topic, info) in enumerate(meta.get("cameras", {}).items(), start=1):
        ax = fig.add_subplot(gs[0, col])
        img = cameras.get(topic)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
        offset = info.get("offset_ms", "N/A") if info else "N/A"
        ax.set_title(f"{topic.split('/')[-1]}\noffset={offset} ms", fontsize=8)
        ax.axis("off")

    # 2. Pointcloud top-down view (x-y)
    pts = pc
    if len(pts) > max_points:
        pts = pts[random.sample(range(len(pts)), max_points)]

    ax_top = fig.add_subplot(gs[1, 0])
    ax_top.scatter(pts[:, 0], pts[:, 1], s=0.5, c=pts[:, 2], cmap="viridis")
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title(f"Pointcloud Top-Down (n={pc.shape[0]})")
    ax_top.set_aspect("equal", adjustable="box")

    # 3. Pointcloud side view (x-z)
    ax_side = fig.add_subplot(gs[1, 1])
    ax_side.scatter(pts[:, 0], pts[:, 2], s=0.5, c=pts[:, 1], cmap="plasma")
    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Z (m)")
    ax_side.set_title("Pointcloud Side View")

    # 4. Metadata / offsets table
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis("off")

    rows = [
        ["Ref topic", meta["ref_topic"]],
        ["Ref timestamp (ns)", str(meta["ref_timestamp_ns"])],
        ["Ref time", meta["ref_timestamp_iso"]],
    ]
    for topic, info in meta.items():
        if topic.startswith("/") and isinstance(info, dict):
            offset = info.get("offset_ms", "N/A")
            ts = info.get("timestamp_ns", "N/A")
            rows.append([topic, f"offset={offset} ms, ts={ts}"])

    table = ax_meta.table(
        cellText=rows,
        colLabels=["Field", "Value"],
        loc="center",
        cellLoc="left",
        colColours=["#eeeeee"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax_meta.set_title("Synchronization Metadata", y=0.95)

    fig.suptitle(f"Synced Sample: {sample_dir.name}", fontsize=14)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def plot_offset_distribution(
    index: dict,
    output_path: Path,
) -> None:
    """Plot histogram of absolute time offsets per synchronized topic and camera."""
    camera_topics = [index["ref_topic"]] + list(index.get("extra_camera_topics", []))
    sensor_topics = list(index.get("sync_topics", {}).keys())
    all_topics = camera_topics + sensor_topics
    offsets: dict[str, list[float]] = {t: [] for t in all_topics}

    for sample in index["samples"]:
        # reference camera
        ref_info = {"offset_ms": 0.0}
        offsets[index["ref_topic"]].append(0.0)
        # extra cameras
        for topic in index.get("extra_camera_topics", []):
            info = sample.get("cameras", {}).get(topic)
            if isinstance(info, dict) and "offset_ms" in info:
                offsets[topic].append(abs(info["offset_ms"]))
        # sensors
        for topic in sensor_topics:
            info = sample.get(topic)
            if isinstance(info, dict) and "offset_ms" in info:
                offsets[topic].append(abs(info["offset_ms"]))

    n_topics = len(all_topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, topic in enumerate(all_topics):
        ax = axes[idx]
        data = offsets[topic]
        ax.hist(data, bins=50, edgecolor="black")
        ax.set_title(topic, fontsize=9)
        ax.set_xlabel("Abs offset (ms)")
        ax.set_ylabel("Count")
        mean = np.mean(data) if data else 0
        max_val = np.max(data) if data else 0
        ax.axvline(mean, color="r", linestyle="--")
        ax.text(
            0.95,
            0.95,
            f"mean={mean:.2f} ms\nmax={max_val:.2f} ms",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Hide unused subplots
    for idx in range(n_topics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Time Synchronization Offsets (Cameras + Sensors)", fontsize=14)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved offset distribution to {output_path}")


def plot_imu_data(
    sample_dir: Path,
    output_path: Path,
) -> None:
    """Plot angular velocity and linear acceleration for both IMUs."""
    imu_topics = ["/camera/camera/imu", "/livox_Lidar_node/imu/raw_data"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, topic in enumerate(imu_topics):
        safe_name = topic.strip("/").replace("/", "_")
        json_path = sample_dir / f"{safe_name}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            imu = json.load(f)

        angular = [
            imu["angular_velocity_x"],
            imu["angular_velocity_y"],
            imu["angular_velocity_z"],
        ]
        linear = [
            imu["linear_acceleration_x"],
            imu["linear_acceleration_y"],
            imu["linear_acceleration_z"],
        ]

        ax_ang = axes[0, col]
        ax_ang.bar(["x", "y", "z"], angular, color=["r", "g", "b"], alpha=0.7)
        ax_ang.set_title(f"{topic}\nAngular Velocity (rad/s)")
        ax_ang.set_ylabel("rad/s")
        ax_ang.grid(True, alpha=0.3)

        ax_lin = axes[1, col]
        ax_lin.bar(["x", "y", "z"], linear, color=["r", "g", "b"], alpha=0.7)
        ax_lin.set_title(f"{topic}\nLinear Acceleration (m/s²)")
        ax_lin.set_ylabel("m/s²")
        ax_lin.grid(True, alpha=0.3)

    fig.suptitle(f"IMU Data for {sample_dir.name}", fontsize=14)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved IMU plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synced VLA samples")
    parser.add_argument(
        "--sync-dir",
        type=str,
        default=DEFAULT_SYNC_DIR,
        help="Path to synced_vla_samples directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Index of sample to visualize (from index.json)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random sample to visualize",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=None,
        help="Number of random samples to visualize",
    )
    parser.add_argument(
        "--plot-offsets",
        action="store_true",
        help="Plot time offset distribution for all samples",
    )
    parser.add_argument(
        "--plot-imu",
        action="store_true",
        help="Plot IMU data for the selected sample",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Max number of points to render for pointcloud",
    )
    args = parser.parse_args()

    sync_dir = Path(args.sync_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    index = load_index(sync_dir)
    print(f"Loaded {index['num_samples']} synced samples")

    if args.plot_offsets:
        plot_offset_distribution(index, output_dir / "offset_distribution.png")

    # Determine samples to visualize
    sample_indices: list[int] = []
    if args.sample_index is not None:
        sample_indices = [args.sample_index]
    elif args.num_random is not None:
        n = min(args.num_random, index["num_samples"])
        sample_indices = random.sample(range(index["num_samples"]), n)
    elif args.random:
        sample_indices = [random.randint(0, index["num_samples"] - 1)]

    if sample_indices:
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= index["num_samples"]:
                print(f"Sample index {sample_idx} out of range [0, {index['num_samples']})")
                continue

            sample_info = index["samples"][sample_idx]
            ref_ts = sample_info["ref_timestamp_ns"]
            sample_dir = sync_dir / f"sample_{ref_ts}"
            print(f"Visualizing sample {sample_idx}: {sample_dir.name}")

            visualize_single_sample(
                sample_dir,
                output_dir / f"sample_{ref_ts}.png",
                max_points=args.max_points,
            )

            if args.plot_imu:
                plot_imu_data(sample_dir, output_dir / f"sample_{ref_ts}_imu.png")

    elif not args.plot_offsets:
        print("Nothing to do. Use --sample-index, --random, --num-random, or --plot-offsets.")


if __name__ == "__main__":
    main()
