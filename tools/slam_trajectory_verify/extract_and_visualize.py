#!/usr/bin/env python3
"""
Extract color camera images from an MCAP bag and overlay the SLAM trajectory.

For each camera frame inside the trajectory time window, find the nearest SLAM
base_link pose and draw a side-by-side view:
  - left: camera image
  - right: 2D top-down trajectory (x, y) with current pose highlighted

Outputs are saved frame-by-frame to --frames-output-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# mcap_processor lives at tools/mcap_processor/src
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_MCAP_SRC = _PROJECT_ROOT / "tools" / "mcap_processor" / "src"
if str(_MCAP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCAP_SRC))

from mcap_processor import BagReader


def load_trajectory(json_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SLAM trajectory and return (timestamps, positions, quaternions)."""
    with open(json_path, "r") as f:
        records = json.load(f)
    records = sorted(records, key=lambda r: r["timestamp"])
    timestamps = np.array([r["timestamp"] for r in records], dtype=np.float64)
    positions = np.array(
        [[r["pose"]["x"], r["pose"]["y"], r["pose"]["z"]] for r in records],
        dtype=np.float64,
    )
    quaternions = np.array(
        # JSON stores [w, x, y, z]
        [[r["pose"]["quaternion"][0], r["pose"]["quaternion"][1],
          r["pose"]["quaternion"][2], r["pose"]["quaternion"][3]] for r in records],
        dtype=np.float64,
    )
    return timestamps, positions, quaternions


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def transform_to_initial_body_frame(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """
    Transform odometry positions into the initial base_link body frame.
    In the body frame: x = forward, y = left, z = up (right-handed).
    """
    p0 = positions[0]
    R0 = quaternion_to_rotation_matrix(quaternions[0])
    # Body-frame coordinates: p_body = R0.T @ (p_odom - p0)
    return (R0.T @ (positions - p0).T).T


def extract_images_in_window(
    bag_dir: str,
    topic: str,
    start_time: float,
    end_time: float,
    image_output_dir: Path,
) -> list[tuple[float, Path]]:
    """Extract color camera images within [start_time, end_time] seconds."""
    image_output_dir.mkdir(parents=True, exist_ok=True)
    reader = BagReader(bag_dir)
    start_ns = int(start_time * 1e9)
    end_ns = int(end_time * 1e9)

    frames: list[tuple[float, Path]] = []
    for msg in reader.iter_messages(topics=[topic], start_time_ns=start_ns, end_time_ns=end_ns):
        ros_msg = msg.ros_msg
        height, width = ros_msg.height, ros_msg.width
        encoding = ros_msg.encoding.lower()

        if encoding in ("rgb8", "rgba8"):
            channels = 4 if "a" in encoding else 3
            img = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape((height, width, channels))
            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding in ("bgr8", "bgra8"):
            channels = 4 if "a" in encoding else 3
            img = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape((height, width, channels))
        elif encoding in ("mono8", "8uc1"):
            img = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape((height, width))
        else:
            print(f"Skipping unsupported encoding {encoding} @ {msg.log_time_ns}")
            continue

        ts = msg.log_time_ns / 1e9
        filename = image_output_dir / f"{msg.log_time_ns:020d}.png"
        cv2.imwrite(str(filename), img)
        frames.append((ts, filename))

    print(f"Extracted {len(frames)} images from {topic} to {image_output_dir}")
    return sorted(frames)


def find_nearest_pose(
    image_ts: float,
    traj_ts: np.ndarray,
    positions: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Return (index, time_offset, position) of nearest trajectory pose."""
    idx = int(np.argmin(np.abs(traj_ts - image_ts)))
    return idx, traj_ts[idx] - image_ts, positions[idx]


def compute_future_trajectory_in_current_frame(
    current_idx: int,
    traj_ts: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    horizon_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute future trajectory points within `horizon_s` seconds in the current
    base_link body frame. Returns (x_forward, y_left) arrays or None if no future.
    """
    current_ts = traj_ts[current_idx]
    end_ts = current_ts + horizon_s

    # Find future indices within horizon
    future_mask = (traj_ts > current_ts) & (traj_ts <= end_ts)
    if not future_mask.any():
        return None

    future_indices = np.where(future_mask)[0]
    p_current = positions[current_idx]
    q_current = quaternions[current_idx]
    R_current = quaternion_to_rotation_matrix(q_current)

    # Transform future positions to current body frame
    future_odom = positions[future_indices]
    future_body = (R_current.T @ (future_odom - p_current).T).T

    # body frame: x=forward, y=left
    return future_body[:, 0], future_body[:, 1]


def compute_future_plot_limits(
    frames: list[tuple[float, Path]],
    traj_ts: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    horizon_s: float,
    margin_ratio: float = 0.15,
) -> tuple[float, float, float, float]:
    """
    Compute fixed x/y limits for the future-trajectory subplot across all frames.
    Returns (y_min, y_max, x_min, x_max) for the right subplot where:
      - horizontal axis is y (left)
      - vertical axis is x (forward)
    """
    y_max_abs = 0.5
    x_max_abs = 0.5
    for ts, _ in frames:
        idx = int(np.argmin(np.abs(traj_ts - ts)))
        future = compute_future_trajectory_in_current_frame(
            idx, traj_ts, positions, quaternions, horizon_s=horizon_s
        )
        if future is not None:
            x_forward, y_left = future
            if len(x_forward) > 0:
                x_max_abs = max(x_max_abs, np.max(np.abs(x_forward)))
                y_max_abs = max(y_max_abs, np.max(np.abs(y_left)))

    x_margin = max(x_max_abs * margin_ratio, 0.3)
    y_margin = max(y_max_abs * margin_ratio, 0.3)
    return (
        -y_max_abs - y_margin,
        y_max_abs + y_margin,
        -x_margin,
        x_max_abs + x_margin,
    )


def render_frame(
    image_path: Path,
    image_ts: float,
    traj_ts: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    body_poses: np.ndarray,
    plot_limits: tuple[float, float, float, float],
    future_plot_limits: tuple[float, float, float, float],
    frame_idx: int,
    output_path: Path,
    future_horizon_s: float = 3.0,
) -> None:
    """Render one frame: camera image + global trajectory + future local trajectory."""
    idx, offset, current_body_pose = find_nearest_pose(image_ts, traj_ts, body_poses)

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Left: camera image
    axes[0].imshow(img)
    axes[0].set_title(f"Camera @ {image_ts:.6f}s (frame {frame_idx})")
    axes[0].axis("off")

    # Middle: full top-down trajectory with fixed axes
    x_min, x_max, y_min, y_max = plot_limits

    # Full trajectory in light blue/gray
    axes[1].plot(body_poses[:, 0], body_poses[:, 1], "-", color="#7FB3D5", linewidth=1.5, label="trajectory")

    # Start / end markers
    axes[1].scatter(
        body_poses[0, 0],
        body_poses[0, 1],
        c="green",
        s=120,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="start",
    )
    axes[1].scatter(
        body_poses[-1, 0],
        body_poses[-1, 1],
        c="red",
        s=120,
        marker="X",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="end",
    )

    # Current pose: moving green dot
    axes[1].scatter(
        current_body_pose[0],
        current_body_pose[1],
        c="lime",
        s=180,
        marker="o",
        edgecolors="darkgreen",
        linewidths=2.0,
        zorder=6,
        label="current",
    )

    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title(
        f"SLAM trajectory in initial base_link frame\n"
        f"x=forward, y=left | current=({current_body_pose[0]:.2f}, {current_body_pose[1]:.2f}) | "
        f"t-offset={offset*1000:.1f}ms"
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # Right: future trajectory in current base_link frame
    # x=forward points UP, y=left points LEFT (first-person view)
    future = compute_future_trajectory_in_current_frame(
        idx, traj_ts, positions, quaternions, horizon_s=future_horizon_s
    )

    if future is not None:
        x_forward, y_left = future
        axes[2].plot(y_left, x_forward, "b-", linewidth=2.0, label=f"future {future_horizon_s}s")
        axes[2].plot(
            y_left[::max(1, len(y_left)//20)],
            x_forward[::max(1, len(y_left)//20)],
            "o",
            color="cornflowerblue",
            markersize=4,
            zorder=4,
        )

    # Current robot position at origin
    axes[2].scatter(
        0.0,
        0.0,
        c="lime",
        s=200,
        marker="o",
        edgecolors="darkgreen",
        linewidths=2.5,
        zorder=6,
        label="current",
    )

    # Draw forward/left coordinate axes at fixed positions inside the plot
    y_min, y_max, x_min, x_max = future_plot_limits
    arrow_len_x = (x_max - x_min) * 0.15
    arrow_len_y = (y_max - y_min) * 0.15
    axes[2].annotate(
        "forward (+x)",
        xy=(0, 0),
        xytext=(0, arrow_len_x),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=9,
        ha="center",
    )
    axes[2].annotate(
        "left (+y)",
        xy=(0, 0),
        xytext=(-arrow_len_y, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=9,
        ha="right",
        va="center",
    )

    # Fixed axes so the coordinate system does not jump between frames
    y_min, y_max, x_min, x_max = future_plot_limits
    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(x_min, x_max)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].invert_xaxis()  # positive y (left) on the left side of the plot
    axes[2].set_xlabel("y (left)")
    axes[2].set_ylabel("x (forward)")
    axes[2].set_title(
        f"Future {future_horizon_s}s trajectory in current base_link frame\n"
        f"x=forward (up), y=left (left)"
    )
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.12, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def compute_plot_limits(poses: np.ndarray, margin_ratio: float = 0.1) -> tuple[float, float, float, float]:
    """Compute fixed x/y limits for the trajectory plot with a margin."""
    x_min, x_max = poses[:, 0].min(), poses[:, 0].max()
    y_min, y_max = poses[:, 1].min(), poses[:, 1].max()
    x_margin = max(abs(x_max - x_min) * margin_ratio, 0.5)
    y_margin = max(abs(y_max - y_min) * margin_ratio, 0.5)
    return (
        x_min - x_margin,
        x_max + x_margin,
        y_min - y_margin,
        y_max + y_margin,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract images and overlay SLAM trajectory")
    parser.add_argument("--bag-dir", type=str, required=True, help="Path to MCAP bag directory")
    parser.add_argument("--traj-json", type=str, required=True, help="Path to SLAM trajectory JSON")
    parser.add_argument("--camera-topic", type=str, default="/camera/camera/color/image_raw")
    parser.add_argument("--image-output-dir", type=str, default="outputs/slam_trajectory_verify/images")
    parser.add_argument("--frames-output-dir", type=str, default="outputs/slam_trajectory_verify/frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to render")
    parser.add_argument("--future-horizon", type=float, default=3.0, help="Future trajectory horizon in seconds")
    args = parser.parse_args()

    matplotlib.use("Agg")

    print("[1/3] Loading trajectory...")
    traj_ts, positions, quaternions = load_trajectory(args.traj_json)
    print(f"  raw trajectory: {len(traj_ts)} poses, {traj_ts[-1] - traj_ts[0]:.1f}s")
    print("  transforming odometry poses into initial base_link body frame (x=forward, y=left)...")
    poses = transform_to_initial_body_frame(positions, quaternions)
    print(f"  body-frame start: ({poses[0, 0]:.2f}, {poses[0, 1]:.2f})")
    print(f"  body-frame end:   ({poses[-1, 0]:.2f}, {poses[-1, 1]:.2f})")
    plot_limits = compute_plot_limits(poses)

    print("[2/3] Extracting camera images...")
    frames = extract_images_in_window(
        args.bag_dir,
        args.camera_topic,
        float(traj_ts[0]),
        float(traj_ts[-1]),
        Path(args.image_output_dir),
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    print("[2.5/3] Computing fixed future-trajectory plot limits...")
    future_plot_limits = compute_future_plot_limits(
        frames, traj_ts, positions, quaternions, horizon_s=args.future_horizon
    )
    print(f"  future plot limits (y_left, x_forward): {future_plot_limits}")

    print(f"[3/3] Rendering {len(frames)} frames...")
    frames_dir = Path(args.frames_output_dir)
    for i, (ts, image_path) in enumerate(frames, 1):
        out_path = frames_dir / f"frame_{i:05d}_{int(ts*1e9):020d}.png"
        render_frame(
            image_path,
            ts,
            traj_ts,
            positions,
            quaternions,
            poses,
            plot_limits,
            future_plot_limits,
            i,
            out_path,
            future_horizon_s=args.future_horizon,
        )
        if i % 50 == 0:
            print(f"  rendered {i}/{len(frames)}")

    print(f"Done. Frames saved to {frames_dir}")


if __name__ == "__main__":
    main()
