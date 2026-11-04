#!/usr/bin/env python3
"""Visualize a single Dobot VLA training sample.

Shows the latest history image and the future trajectory in the ego frame.
Coordinate convention for the plot:
  - upward  = ego x (forward)
  - leftward = ego y (left)

Usage:
    python tools/vis_dobot_vla_sample.py \
        --json_list /root/embodvla_zhouyuan/outputs/post_sync_data/all_training_jsons.txt \
        --idx 0 \
        --output /tmp/dobot_vla_sample.png
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from vla.data.dobot_vla import (
    DobotVLADataset,
    find_nearest_frame_idx,
    interpolate_pose,
    transform_to_ego,
)
from vla.utils.common import build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_list", type=str, required=True,
                        help="Path to txt file listing training.json paths.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of the sample to visualize.")
    parser.add_argument("--output", type=str, default="/tmp/dobot_vla_sample.png",
                        help="Output image path.")
    parser.add_argument("--model_path", type=str,
                        default="/root/embodvla/models/OpenGVLab/InternVL2_5-1B",
                        help="Tokenizer path.")
    parser.add_argument("--cam_channel", type=str, default="cam1")
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--history_duration", type=float, default=1.0)
    parser.add_argument("--n_future", type=int, default=24)
    parser.add_argument("--future_dt", type=float, default=0.25)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Only use the first N videos for faster visualization.")
    parser.add_argument("--traj_xlim", type=float, nargs=2, default=[-5.0, 5.0],
                        help="Fixed x-axis limits for trajectory plot (ego y, left-right).")
    parser.add_argument("--traj_ylim", type=float, nargs=2, default=[-1.0, 6.0],
                        help="Fixed y-axis limits for trajectory plot (ego x, backward-forward).")
    return parser.parse_args()


def denormalize(tensor: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = np.transpose(img, (1, 2, 0))
    return img


def draw_arrow(ax, x, y, yaw, length, color, linewidth=1.5):
    """Draw an arrow representing heading (yaw) in ego frame.

    In ego frame:
      - forward (x) is up on the plot
      - left (y) is to the left on the plot
    """
    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    # Map ego (x, y) -> plot (left=-y, up=x)
    ax.annotate(
        "",
        xy=(-(y + dy), x + dx),
        xytext=(-y, x),
        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth),
    )


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    with open(args.json_list) as f:
        json_paths = [line.strip() for line in f if line.strip()]
    if args.max_videos is not None:
        json_paths = json_paths[:args.max_videos]
        print(f"Using first {len(json_paths)} videos for visualization.")

    transform = build_transform(args.input_size)
    dataset = DobotVLADataset(
        json_paths=json_paths,
        tokenizer=tokenizer,
        transform=transform,
        num_history_frames=args.num_history_frames,
        history_duration=args.history_duration,
        n_future=args.n_future,
        future_dt=args.future_dt,
        cam_channel=args.cam_channel,
        sample_interval=args.sample_interval,
        cache_dir=None,  # no cache for vis
    )

    print(f"Dataset size: {len(dataset)}")
    idx = max(0, min(args.idx, len(dataset) - 1))
    sample = dataset[idx]
    traj = sample["trajectory"].numpy()  # (T, 3) [x, y, yaw]
    history_pixel_values = sample["pixel_values"].numpy()  # (H, C, H, W)

    # Latest history image is the last one.
    latest_img_tensor = history_pixel_values[-1]
    latest_img = denormalize(latest_img_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: history image.
    ax_img = axes[0]
    ax_img.imshow(latest_img)
    ax_img.set_title(f"Latest history frame ({args.cam_channel})\n{sample['token']}")
    ax_img.axis("off")

    # Right: trajectory in ego frame.
    ax_traj = axes[1]
    ax_traj.set_aspect("equal", adjustable="box")

    # Plot convention: upward = ego x (forward), leftward = ego y (left).
    # Map ego (x, y) -> plot coordinates (px=-y, py=x).
    plot_x = -traj[:, 1]
    plot_y = traj[:, 0]

    ax_traj.plot(plot_x, plot_y, "b-o", markersize=4, linewidth=1.5, label="future trajectory")
    ax_traj.plot(0, 0, "r*", markersize=15, label="current ego")

    # Draw heading arrows for every 4th point + last point.
    arrow_indices = list(range(0, len(traj), 4)) + [len(traj) - 1]
    for i in sorted(set(arrow_indices)):
        draw_arrow(ax_traj, traj[i, 0], traj[i, 1], traj[i, 2], length=0.15, color="green")

    # Draw ego heading arrow at origin.
    draw_arrow(ax_traj, 0.0, 0.0, 0.0, length=0.25, color="red", linewidth=2.0)

    ax_traj.set_xlim(args.traj_xlim)
    ax_traj.set_ylim(args.traj_ylim)
    ax_traj.set_xlabel("ego y (left)")
    ax_traj.set_ylabel("ego x (forward)")
    ax_traj.set_title("Future trajectory in ego frame\n(up=x, left=y)")
    ax_traj.legend(loc="upper left")
    ax_traj.grid(True, linestyle="--", alpha=0.5)

    # Add text annotation for trajectory stats.
    stats_text = (
        f"n_future: {args.n_future}\n"
        f"future_dt: {args.future_dt}s\n"
        f"final (x,y): ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})\n"
        f"final yaw: {traj[-1, 2]:.2f} rad"
    )
    ax_traj.text(
        0.98, 0.02, stats_text,
        transform=ax_traj.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
