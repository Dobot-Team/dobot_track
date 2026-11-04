#!/usr/bin/env python3
"""Render a visualization video for one Dobot training.json.

For every valid label in the video, the script renders a frame showing:
  - left: current cam1 image
  - right: future trajectory in the ego frame (up=x/forward, left=y/left)

The output is an MP4 video at the label frequency (10 fps by default).

Usage:
    python tools/vis_dobot_vla_video.py \
        --json /mnt/oss/post_sync_data/20260630/N-01-035-004-1782811378283823033/dobot_bag_0000_0/training.json \
        --output /tmp/dobot_vla_video.mp4 \
        --fps 10
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vla.data.dobot_vla import (
    find_nearest_frame_idx,
    interpolate_pose,
    transform_to_ego,
)
from vla.utils.common import build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to training.json.")
    parser.add_argument("--output", type=str, default="/tmp/dobot_vla_video.mp4",
                        help="Output MP4 path.")
    parser.add_argument("--cam_channel", type=str, default="cam1")
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--history_duration", type=float, default=1.0)
    parser.add_argument("--n_future", type=int, default=24)
    parser.add_argument("--future_dt", type=float, default=0.25)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video fps. Labels are ~10 Hz.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Render at most N frames (for quick test).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Render one frame every N labels.")
    parser.add_argument("--traj_xlim", type=float, nargs=2, default=[-5.0, 5.0],
                        help="Fixed x-axis limits for trajectory plot (ego y, left-right).")
    parser.add_argument("--traj_ylim", type=float, nargs=2, default=[-1.0, 6.0],
                        help="Fixed y-axis limits for trajectory plot (ego x, backward-forward).")
    return parser.parse_args()


def load_json_data(json_path: str):
    import json
    with open(json_path) as f:
        data = json.load(f)
    frames = data.get("frames", [])
    labels = data.get("labels", [])
    return frames, labels


def denormalize(tensor: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = np.transpose(img, (1, 2, 0))
    return img


def draw_arrow(ax, x, y, yaw, length, color, linewidth=1.5):
    """Draw an arrow representing heading (yaw) in ego frame.

    Plot convention: ego_x -> upward, ego_y -> leftward.
    """
    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    ax.annotate(
        "",
        xy=(-(y + dy), x + dx),
        xytext=(-y, x),
        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth),
    )


def render_frame(
    current_image: np.ndarray,
    traj: np.ndarray,
    frame_label: str,
    args,
) -> np.ndarray:
    """Render a single visualization frame as RGB uint8 array."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: current image.
    ax_img = axes[0]
    ax_img.imshow(current_image)
    ax_img.set_title(f"Current frame ({args.cam_channel})\n{frame_label}")
    ax_img.axis("off")

    # Right: trajectory in ego frame.
    ax_traj = axes[1]
    ax_traj.set_aspect("equal", adjustable="box")

    plot_x = -traj[:, 1]
    plot_y = traj[:, 0]

    ax_traj.plot(plot_x, plot_y, "b-o", markersize=3, linewidth=1.2, label="future trajectory")
    ax_traj.plot(0, 0, "r*", markersize=12, label="current ego")

    # Heading arrows every few points + last point.
    arrow_indices = list(range(0, len(traj), 6)) + [len(traj) - 1]
    for i in sorted(set(arrow_indices)):
        draw_arrow(ax_traj, traj[i, 0], traj[i, 1], traj[i, 2], length=0.12, color="green")
    draw_arrow(ax_traj, 0.0, 0.0, 0.0, length=0.20, color="red", linewidth=2.0)

    ax_traj.set_xlim(args.traj_xlim)
    ax_traj.set_ylim(args.traj_ylim)
    ax_traj.set_xlabel("ego y (left)")
    ax_traj.set_ylabel("ego x (forward)")
    ax_traj.set_title("Future trajectory in ego frame\n(up=x, left=y)")
    ax_traj.legend(loc="upper left", fontsize=8)
    ax_traj.grid(True, linestyle="--", alpha=0.5)

    stats_text = (
        f"n_future: {args.n_future}\n"
        f"future_dt: {args.future_dt}s\n"
        f"final (x,y): ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})\n"
        f"final yaw: {traj[-1, 2]:.2f} rad"
    )
    ax_traj.text(
        0.98, 0.02, stats_text,
        transform=ax_traj.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return frame


def main():
    args = parse_args()

    print(f"Loading {args.json} ...")
    frames, labels = load_json_data(args.json)
    print(f"  frames: {len(frames)}, labels: {len(labels)}")

    if len(frames) == 0 or len(labels) == 0:
        print("Empty data, exiting.")
        return

    frame_ts = np.array([f["timestamp"] * 1e-9 for f in frames], dtype=np.float64)
    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)

    history_offsets = np.linspace(-args.history_duration, 0.0, args.num_history_frames)
    future_offsets = np.arange(1, args.n_future + 1) * args.future_dt

    transform = build_transform(args.input_size)

    # Collect valid label indices.
    valid_indices = []
    for i in range(0, len(labels), args.stride):
        curr_t = label_ts[i]
        if curr_t + history_offsets[0] < frame_ts[0]:
            continue
        if curr_t + future_offsets[-1] > label_ts[-1]:
            continue
        valid_indices.append(i)

    if args.max_frames is not None:
        valid_indices = valid_indices[:args.max_frames]

    print(f"Rendering {len(valid_indices)} frames at {args.fps} fps -> "
          f"~{len(valid_indices)/args.fps:.1f}s video")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Determine video size from first rendered frame.
    sample_label = labels[valid_indices[0]]
    curr_t = sample_label["timestamp"]
    curr_pose = np.array([
        sample_label["pose"]["x"],
        sample_label["pose"]["y"],
        sample_label["pose"]["yaw"],
    ], dtype=np.float32)

    # Current image.
    history_ts = curr_t + history_offsets
    history_frame_idxs = [find_nearest_frame_idx(t, frame_ts) for t in history_ts]
    current_frame_idx = history_frame_idxs[-1]
    current_img_path = frames[current_frame_idx][args.cam_channel]
    current_img = Image.open(current_img_path).convert("RGB")
    current_img_tensor = transform(current_img).numpy()
    current_img_vis = denormalize(current_img_tensor)

    # Future trajectory.
    future_ts = curr_t + future_offsets
    future_poses = np.array([interpolate_pose(t, labels) for t in future_ts], dtype=np.float32)
    traj = transform_to_ego(curr_pose, future_poses)

    first_frame = render_frame(current_img_vis, traj, f"frame {valid_indices[0]}", args)
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    # Write first frame.
    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    for idx in valid_indices[1:]:
        label = labels[idx]
        curr_t = label["timestamp"]
        curr_pose = np.array([
            label["pose"]["x"],
            label["pose"]["y"],
            label["pose"]["yaw"],
        ], dtype=np.float32)

        history_ts = curr_t + history_offsets
        history_frame_idxs = [find_nearest_frame_idx(t, frame_ts) for t in history_ts]
        current_frame_idx = history_frame_idxs[-1]
        current_img_path = frames[current_frame_idx][args.cam_channel]

        try:
            current_img = Image.open(current_img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping frame {idx}: {e}")
            continue

        current_img_tensor = transform(current_img).numpy()
        current_img_vis = denormalize(current_img_tensor)

        future_ts = curr_t + future_offsets
        future_poses = np.array([interpolate_pose(t, labels) for t in future_ts], dtype=np.float32)
        traj = transform_to_ego(curr_pose, future_poses)

        frame = render_frame(current_img_vis, traj, f"frame {idx}", args)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
