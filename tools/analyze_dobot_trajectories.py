#!/usr/bin/env python3
"""Analyze all Dobot trajectories with a sliding window over future 3s.

For every valid label in every training.json, compute the future 3s trajectory
in the ego frame and output statistics + plots.

Usage:
    python tools/analyze_dobot_trajectories.py \
        --json_list /mnt/cpfs/dobot_vla_local/all_training_jsons_local.txt \
        --output_dir /tmp/dobot_traj_analysis \
        --future_seconds 3.0 \
        --future_dt 0.25
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/dobot_traj_analysis")
    parser.add_argument("--future_seconds", type=float, default=3.0)
    parser.add_argument("--future_dt", type=float, default=0.25,
                        help="Sampling interval for future trajectory points.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_videos", type=int, default=None)
    return parser.parse_args()


def load_json(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def transform_to_ego(current_pose: np.ndarray, future_poses: np.ndarray) -> np.ndarray:
    """Transform future poses into the current ego frame.

    ego frame: x=forward, y=left, yaw=relative to current heading.
    """
    dx = future_poses[:, 0] - current_pose[0]
    dy = future_poses[:, 1] - current_pose[1]
    cos_yaw = np.cos(current_pose[2])
    sin_yaw = np.sin(current_pose[2])

    ego_x = dx * cos_yaw + dy * sin_yaw
    ego_y = -dx * sin_yaw + dy * cos_yaw
    ego_yaw = future_poses[:, 2] - current_pose[2]
    ego_yaw = (ego_yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack([ego_x, ego_y, ego_yaw], axis=-1).astype(np.float32)


def interpolate_pose(t: float, labels: List[dict]) -> np.ndarray:
    ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)
    idx = int(np.searchsorted(ts, t))
    if idx == 0:
        p = labels[0]["pose"]
        return np.array([p["x"], p["y"], p["yaw"]], dtype=np.float32)
    if idx >= len(labels):
        p = labels[-1]["pose"]
        return np.array([p["x"], p["y"], p["yaw"]], dtype=np.float32)

    t0, t1 = ts[idx - 1], ts[idx]
    alpha = float((t - t0) / (t1 - t0)) if t1 != t0 else 0.0

    p0 = labels[idx - 1]["pose"]
    p1 = labels[idx]["pose"]
    x = (1.0 - alpha) * p0["x"] + alpha * p1["x"]
    y = (1.0 - alpha) * p0["y"] + alpha * p1["y"]

    y0, y1 = p0["yaw"], p1["yaw"]
    dy = (y1 - y0 + np.pi) % (2.0 * np.pi) - np.pi
    yaw = y0 + alpha * dy
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([x, y, yaw], dtype=np.float32)


def analyze_video(args: Tuple[str, float, float]) -> dict:
    json_path, future_seconds, future_dt = args
    try:
        data = load_json(json_path)
    except Exception as e:
        return {"error": str(e), "num_traj": 0}

    labels = data.get("labels", [])
    if len(labels) < 2:
        return {"num_traj": 0}

    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)
    n_future = int(round(future_seconds / future_dt))
    future_offsets = np.arange(1, n_future + 1) * future_dt
    last_label_t = label_ts[-1]

    trajs = []  # list of (n_future, 3) ego trajectories
    for i in range(len(labels)):
        curr_t = label_ts[i]
        if curr_t + future_offsets[-1] > last_label_t:
            continue

        curr_label = labels[i]
        curr_pose = np.array([
            curr_label["pose"]["x"],
            curr_label["pose"]["y"],
            curr_label["pose"]["yaw"],
        ], dtype=np.float32)

        future_ts = curr_t + future_offsets
        future_poses = np.array([interpolate_pose(t, labels) for t in future_ts], dtype=np.float32)
        traj_ego = transform_to_ego(curr_pose, future_poses)
        trajs.append(traj_ego)

    if len(trajs) == 0:
        return {"num_traj": 0}

    trajs = np.stack(trajs, axis=0)  # (N, T, 3)

    # Per-trajectory features.
    final_x = trajs[:, -1, 0]
    final_y = trajs[:, -1, 1]
    final_yaw = trajs[:, -1, 2]
    max_abs_y = np.max(np.abs(trajs[:, :, 1]), axis=1)
    max_abs_yaw = np.max(np.abs(trajs[:, :, 2]), axis=1)
    yaw_change = trajs[:, -1, 2] - trajs[:, 0, 2]
    yaw_change = (yaw_change + np.pi) % (2.0 * np.pi) - np.pi
    path_length = np.sum(np.linalg.norm(np.diff(trajs[:, :, :2], axis=1), axis=-1), axis=1)
    displacement = np.linalg.norm(trajs[:, -1, :2] - trajs[:, 0, :2], axis=1)

    return {
        "num_traj": len(trajs),
        "trajs": trajs,
        "final_x": final_x,
        "final_y": final_y,
        "final_yaw": final_yaw,
        "max_abs_y": max_abs_y,
        "max_abs_yaw": max_abs_yaw,
        "yaw_change": yaw_change,
        "path_length": path_length,
        "displacement": displacement,
    }


def plot_distribution(ax, data, title, xlabel, bins=100, color="steelblue"):
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, linestyle="--", alpha=0.5)
    mean, std = data.mean(), data.std()
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"mean={mean:.3f}")
    ax.axvline(mean + std, color="orange", linestyle="--", linewidth=1, label=f"+1std")
    ax.axvline(mean - std, color="orange", linestyle="--", linewidth=1, label=f"-1std")
    ax.legend(fontsize=8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_list) as f:
        json_paths = [line.strip() for line in f if line.strip()]
    if args.max_videos is not None:
        json_paths = json_paths[:args.max_videos]
    print(f"Analyzing {len(json_paths)} videos ...")

    worker_args = [(p, args.future_seconds, args.future_dt) for p in json_paths]

    all_results = []
    if args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            all_results = list(tqdm(
                pool.imap(analyze_video, worker_args),
                total=len(worker_args),
                desc="Processing videos",
            ))
    else:
        all_results = [analyze_video(a) for a in tqdm(worker_args)]

    # Filter out errors and empty.
    all_results = [r for r in all_results if r.get("num_traj", 0) > 0]
    total_traj = sum(r["num_traj"] for r in all_results)
    print(f"Total trajectories analyzed: {total_traj}")

    if total_traj == 0:
        print("No trajectories found.")
        return

    # Aggregate.
    final_x = np.concatenate([r["final_x"] for r in all_results])
    final_y = np.concatenate([r["final_y"] for r in all_results])
    final_yaw = np.concatenate([r["final_yaw"] for r in all_results])
    max_abs_y = np.concatenate([r["max_abs_y"] for r in all_results])
    max_abs_yaw = np.concatenate([r["max_abs_yaw"] for r in all_results])
    yaw_change = np.concatenate([r["yaw_change"] for r in all_results])
    path_length = np.concatenate([r["path_length"] for r in all_results])
    displacement = np.concatenate([r["displacement"] for r in all_results])

    # Statistics.
    stats = {
        "num_videos": len(all_results),
        "num_trajectories": int(total_traj),
        "future_seconds": args.future_seconds,
        "future_dt": args.future_dt,
        "final_x": {"mean": float(final_x.mean()), "std": float(final_x.std()),
                    "min": float(final_x.min()), "max": float(final_x.max())},
        "final_y": {"mean": float(final_y.mean()), "std": float(final_y.std()),
                    "min": float(final_y.min()), "max": float(final_y.max())},
        "final_yaw": {"mean": float(final_yaw.mean()), "std": float(final_yaw.std()),
                      "min": float(final_yaw.min()), "max": float(final_yaw.max())},
        "max_abs_y": {"mean": float(max_abs_y.mean()), "std": float(max_abs_y.std()),
                      "min": float(max_abs_y.min()), "max": float(max_abs_y.max())},
        "max_abs_yaw": {"mean": float(max_abs_yaw.mean()), "std": float(max_abs_yaw.std()),
                        "min": float(max_abs_yaw.min()), "max": float(max_abs_yaw.max())},
        "yaw_change": {"mean": float(yaw_change.mean()), "std": float(yaw_change.std()),
                       "min": float(yaw_change.min()), "max": float(yaw_change.max())},
        "path_length": {"mean": float(path_length.mean()), "std": float(path_length.std()),
                        "min": float(path_length.min()), "max": float(path_length.max())},
        "displacement": {"mean": float(displacement.mean()), "std": float(displacement.std()),
                         "min": float(displacement.min()), "max": float(displacement.max())},
        "turn_ratios": {
            "straight_or_small_turn": float((np.abs(yaw_change) < 0.2).mean()),
            "moderate_turn_0.2_0.5": float(((np.abs(yaw_change) >= 0.2) & (np.abs(yaw_change) < 0.5)).mean()),
            "large_turn_0.5_1.0": float(((np.abs(yaw_change) >= 0.5) & (np.abs(yaw_change) < 1.0)).mean()),
            "very_large_turn_>=1.0": float((np.abs(yaw_change) >= 1.0).mean()),
        },
        "lateral_motion_ratios": {
            "small_lateral_<=0.1": float((max_abs_y <= 0.1).mean()),
            "moderate_lateral_0.1_0.3": float(((max_abs_y > 0.1) & (max_abs_y <= 0.3)).mean()),
            "large_lateral_0.3_0.6": float(((max_abs_y > 0.3) & (max_abs_y <= 0.6)).mean()),
            "very_large_lateral_>0.6": float((max_abs_y > 0.6).mean()),
        },
    }

    import json as _json
    with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
        _json.dump(stats, f, indent=2)

    print("\nStatistics:")
    for k, v in stats.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}]")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv*100:.1f}%" if isinstance(vv, float) and vv <= 1.0 else f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Plots.
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    plot_distribution(axes[0, 0], final_x, "Final x distribution", "x (forward, m)")
    plot_distribution(axes[0, 1], final_y, "Final y distribution", "y (left, m)")
    plot_distribution(axes[0, 2], final_yaw, "Final yaw distribution", "yaw (rad)")

    plot_distribution(axes[1, 0], max_abs_y, "Max abs lateral y", "max |y| (m)")
    plot_distribution(axes[1, 1], max_abs_yaw, "Max abs yaw", "max |yaw| (rad)")
    plot_distribution(axes[1, 2], yaw_change, "Yaw change (final - initial)", "yaw change (rad)")

    plot_distribution(axes[2, 0], path_length, "Path length", "length (m)")
    plot_distribution(axes[2, 1], displacement, "Displacement", "displacement (m)")

    # Scatter: final x vs final y.
    ax = axes[2, 2]
    sample_idx = np.random.choice(len(final_x), size=min(5000, len(final_x)), replace=False)
    ax.scatter(final_x[sample_idx], final_y[sample_idx], s=1, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("final x (forward, m)")
    ax.set_ylabel("final y (left, m)")
    ax.set_title("Final position scatter (ego frame)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "distributions.png"), dpi=150)
    print(f"\nPlots saved to {args.output_dir}/distributions.png")

    # Per-step statistics.
    all_trajs = np.concatenate([r["trajs"] for r in all_results], axis=0)
    step_x = all_trajs[:, :, 0].mean(axis=0)
    step_y = all_trajs[:, :, 1].mean(axis=0)
    step_yaw = all_trajs[:, :, 2].mean(axis=0)
    step_x_std = all_trajs[:, :, 0].std(axis=0)
    step_y_std = all_trajs[:, :, 1].std(axis=0)
    step_yaw_std = all_trajs[:, :, 2].std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = np.arange(1, all_trajs.shape[1] + 1) * args.future_dt
    axes[0].plot(t, step_x, "b-o", markersize=4, label="mean")
    axes[0].fill_between(t, step_x - step_x_std, step_x + step_x_std, alpha=0.3)
    axes[0].set_title("x over future time")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("x (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t, step_y, "g-o", markersize=4, label="mean")
    axes[1].fill_between(t, step_y - step_y_std, step_y + step_y_std, alpha=0.3)
    axes[1].set_title("y over future time")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("y (m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(t, step_yaw, "r-o", markersize=4, label="mean")
    axes[2].fill_between(t, step_yaw - step_yaw_std, step_yaw + step_yaw_std, alpha=0.3)
    axes[2].set_title("yaw over future time")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("yaw (rad)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_step_stats.png"), dpi=150)
    print(f"Per-step stats saved to {args.output_dir}/per_step_stats.png")


if __name__ == "__main__":
    main()
