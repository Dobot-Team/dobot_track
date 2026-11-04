#!/usr/bin/env python3
"""Build a temporally aligned, command-scaled real TrackVLA dataset.

The existing converted dataset is read-only.  The new manifest selects current
observations at ``current_hz`` and stores an explicit history sampled at
``history_hz``.  Optionally, trajectories are converted from metric robot
motion to a configured command scale:

    normalized forward command = metric forward velocity / forward_speed_scale
    normalized lateral command = metric lateral velocity / lateral_speed_scale
    normalized angular command = metric angular velocity / angular_speed_scale

The scale values must come from the robot controller and must match the units
used by the simulation ``base_velocity`` actions.  They are intentionally
required command-line arguments rather than assumed constants.  The resulting
commands are integrated at ``trajectory_hz`` to form trajectory targets.  RGB
and mask archives can be hard-linked into the new dataset root, avoiding
another full image copy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument(
        "--raw-source-root",
        type=Path,
        default=None,
        help=(
            "Optional pipeline_v2 root. When supplied, reject RGB records whose "
            "timestamps are not aligned to a pose label in the corresponding "
            "training.json. This prevents pre-label camera frames from inheriting "
            "the first trajectory target."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--current-hz", type=float, default=10.0)
    parser.add_argument("--history-hz", type=float, default=5.0)
    parser.add_argument("--history-frames", type=int, default=8)
    parser.add_argument("--trajectory-hz", type=float, default=10.0)
    parser.add_argument(
        "--forward-speed-scale",
        type=float,
        required=True,
        help="Physical forward speed represented by normalized command 1.0 (m/s).",
    )
    parser.add_argument(
        "--lateral-speed-scale",
        type=float,
        required=True,
        help="Physical lateral speed represented by normalized command 1.0 (m/s).",
    )
    parser.add_argument(
        "--angular-speed-scale",
        type=float,
        required=True,
        help="Physical robot speed represented by normalized angular command 1.0 (rad/s).",
    )
    parser.add_argument(
        "--clip-normalized-speed",
        action="store_true",
        help="Clip each normalized vx, vy and omega target to [-1, 1].",
    )
    parser.add_argument(
        "--max-alignment-error-ms",
        type=float,
        default=75.0,
        help="Reject current-frame grid points farther from a camera frame than this value.",
    )
    parser.add_argument(
        "--max-pose-alignment-error-ms",
        type=float,
        default=75.0,
        help="Maximum RGB-to-pose-label timestamp error when --raw-source-root is used.",
    )
    parser.add_argument(
        "--archive-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def iter_json_array(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading it in RAM."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            # Compact only after consuming a sizeable block.  Compacting on
            # every yielded object turns a large manifest into quadratic-time
            # string copying.
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                block = handle.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                position += 1
                started = True

            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"Unterminated JSON array in {path}")
                continue

            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                block = handle.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True
                continue
            if not isinstance(item, dict):
                raise TypeError(f"Expected manifest objects, got {type(item).__name__}")
            position = end
            yield item


def iter_episode_groups(records: Iterable[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    group: List[Dict[str, Any]] = []
    group_key: Any = None
    for record in records:
        key = record.get("frames_file")
        if group and key != group_key:
            yield group
            group = []
        group_key = key
        group.append(record)
    if group:
        yield group


def filter_pose_aligned_records(
    records: List[Dict[str, Any]],
    raw_source_root: Optional[Path],
    tolerance_ns: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop camera records outside the valid pose-label clock.

    The first real-data conversion used unrestricted nearest-neighbour label
    matching.  Camera frames captured before the first pose label therefore
    inherited label zero, sometimes for more than one second.  Recover the
    corresponding raw ``training.json`` from the archive name and enforce a
    bounded timestamp match without touching either source dataset.
    """

    if raw_source_root is None or not records:
        return records, 0
    frames_file = str(records[0].get("frames_file", ""))
    episode_key = Path(frames_file).stem
    parts = episode_key.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Cannot map frames_file to raw training.json: {frames_file!r}"
        )
    training_path = raw_source_root.joinpath(*parts, "training.json")
    if not training_path.is_file():
        raise FileNotFoundError(
            f"Missing raw training metadata for {frames_file}: {training_path}"
        )
    raw = json.loads(training_path.read_text(encoding="utf-8"))
    labels = raw.get("labels", [])
    if not labels:
        raise ValueError(f"No pose labels in {training_path}")
    label_times_ns = [round(float(item["timestamp"]) * 1e9) for item in labels]

    kept: List[Dict[str, Any]] = []
    for record in records:
        timestamp = int(record["source_timestamp"])
        label_index = nearest_index(label_times_ns, timestamp)
        if abs(label_times_ns[label_index] - timestamp) <= tolerance_ns:
            kept.append(record)
    return kept, len(records) - len(kept)


def nearest_index(
    times: Sequence[int],
    target: int,
    upper_bound: Optional[int] = None,
) -> int:
    end = len(times) if upper_bound is None else min(len(times), upper_bound + 1)
    if end <= 0:
        return 0
    position = bisect_left(times, target, 0, end)
    if position <= 0:
        return 0
    if position >= end:
        return end - 1
    before = position - 1
    return position if abs(times[position] - target) < abs(times[before] - target) else before


def normalize_trajectory(
    trajectory: Sequence[Sequence[float]],
    dt: float,
    forward_scale: float,
    lateral_scale: float,
    angular_scale: float,
    clip_speed: bool,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Convert metric relative poses to integrated normalized commands."""

    physical_previous = (0.0, 0.0, 0.0)
    normalized_pose = [0.0, 0.0, 0.0]
    normalized_trajectory: List[List[float]] = []
    normalized_actions: List[List[float]] = []

    for point in trajectory:
        if len(point) < 3:
            raise ValueError(f"Trajectory point must have 3 values, got {point!r}")
        px, py, pyaw = map(float, point[:3])
        prev_x, prev_y, prev_yaw = physical_previous
        dx = px - prev_x
        dy = py - prev_y

        # Initial-ego displacement -> robot-local displacement at the previous
        # trajectory point.
        local_dx = math.cos(prev_yaw) * dx + math.sin(prev_yaw) * dy
        local_dy = -math.sin(prev_yaw) * dx + math.cos(prev_yaw) * dy
        local_dyaw = wrap_angle(pyaw - prev_yaw)

        action = [
            local_dx / (forward_scale * dt),
            local_dy / (lateral_scale * dt),
            local_dyaw / (angular_scale * dt),
        ]
        if clip_speed:
            action = [max(-1.0, min(1.0, value)) for value in action]

        step_x = action[0] * dt
        step_y = action[1] * dt
        step_yaw = action[2] * dt
        heading = normalized_pose[2]
        normalized_pose[0] += math.cos(heading) * step_x - math.sin(heading) * step_y
        normalized_pose[1] += math.sin(heading) * step_x + math.cos(heading) * step_y
        normalized_pose[2] = wrap_angle(normalized_pose[2] + step_yaw)

        normalized_actions.append(action)
        normalized_trajectory.append(normalized_pose.copy())
        physical_previous = (px, py, pyaw)

    return normalized_trajectory, normalized_actions


def resample_group(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], int]:
    records, pose_rejected = filter_pose_aligned_records(
        records,
        args.raw_source_root,
        round(args.max_pose_alignment_error_ms * 1e6),
    )
    records = [record for record in records if record.get("source_timestamp") is not None]
    records.sort(key=lambda record: int(record["source_timestamp"]))
    if not records:
        return [], pose_rejected

    times = [int(record["source_timestamp"]) for record in records]
    current_period = round(1e9 / args.current_hz)
    history_period = round(1e9 / args.history_hz)
    tolerance = round(args.max_alignment_error_ms * 1e6)
    trajectory_dt = 1.0 / args.trajectory_hz

    selected_indices: List[int] = []
    grid_time = times[0]
    previous_index = -1
    while grid_time <= times[-1]:
        index = nearest_index(times, grid_time)
        if index != previous_index and abs(times[index] - grid_time) <= tolerance:
            selected_indices.append(index)
            previous_index = index
        grid_time += current_period

    output: List[Dict[str, Any]] = []
    for current_index in selected_indices:
        current = records[current_index]
        current_time = times[current_index]
        history_names: List[str] = []
        history_valid: List[bool] = []
        for offset in range(args.history_frames - 1, 0, -1):
            target_time = current_time - offset * history_period
            if target_time < times[0]:
                history_index = 0
                valid = False
            else:
                history_index = nearest_index(times, target_time, current_index)
                valid = abs(times[history_index] - target_time) <= tolerance
            history_names.append(str(records[history_index]["current"]))
            history_valid.append(valid)
        history_names.append(str(current["current"]))
        history_valid.append(True)

        normalized_trajectory, normalized_actions = normalize_trajectory(
            current.get("trajectory", []),
            trajectory_dt,
            args.forward_speed_scale,
            args.lateral_speed_scale,
            args.angular_speed_scale,
            args.clip_normalized_speed,
        )
        converted = dict(current)
        converted["history_frames"] = history_names
        converted["history_frame_valid"] = history_valid
        converted["trajectory_metric"] = current.get("trajectory", [])
        converted["trajectory"] = normalized_trajectory
        converted["actions_normalized"] = normalized_actions
        converted["temporal_sampling"] = {
            "current_hz": args.current_hz,
            "history_hz": args.history_hz,
            "trajectory_hz": args.trajectory_hz,
        }
        converted["command_normalization"] = {
            "forward_speed_scale_mps": args.forward_speed_scale,
            "lateral_speed_scale_mps": args.lateral_speed_scale,
            "angular_speed_scale_radps": args.angular_speed_scale,
            "clipped_to_unit_range": args.clip_normalized_speed,
        }
        output.append(converted)
    return output, pose_rejected


def materialize_tree(source: Path, destination: Path, mode: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
        return
    for root, directories, files in os.walk(source):
        relative = Path(root).relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_root / directory).mkdir(exist_ok=True)
        for filename in files:
            src = Path(root) / filename
            dst = target_root / filename
            if mode == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_dataset = args.source_dataset.resolve()
    source_root = source_dataset.parent
    output_root = args.output_root.resolve()
    if not source_dataset.is_file():
        raise FileNotFoundError(source_dataset)
    if args.raw_source_root is not None:
        args.raw_source_root = args.raw_source_root.resolve()
        if not args.raw_source_root.is_dir():
            raise FileNotFoundError(args.raw_source_root)
    for name, value in (
        ("current_hz", args.current_hz),
        ("history_hz", args.history_hz),
        ("trajectory_hz", args.trajectory_hz),
        ("forward_speed_scale", args.forward_speed_scale),
        ("lateral_speed_scale", args.lateral_speed_scale),
        ("angular_speed_scale", args.angular_speed_scale),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if output_root == source_root:
        raise ValueError("output-root must differ from the existing dataset root")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    materialize_tree(source_root / "frames", output_root / "frames", args.archive_mode)
    materialize_tree(source_root / "masks", output_root / "masks", args.archive_mode)

    output_path = output_root / "dataset.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    episode_count = 0
    source_samples = 0
    output_samples = 0
    pose_alignment_rejected = 0
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first = True
        for episode_count, group in enumerate(
            iter_episode_groups(iter_json_array(source_dataset)),
            1,
        ):
            source_samples += len(group)
            converted, rejected = resample_group(group, args)
            pose_alignment_rejected += rejected
            for record in converted:
                if not first:
                    handle.write(",")
                json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                first = False
                output_samples += 1
            if episode_count == 1 or episode_count % 10 == 0:
                print(
                    f"[resample] episodes={episode_count} "
                    f"source_samples={source_samples} output_samples={output_samples} "
                    f"pose_rejected={pose_alignment_rejected}",
                    flush=True,
                )
        handle.write("]")
    os.replace(temporary_path, output_path)

    summary = {
        "source_dataset": str(source_dataset),
        "episodes": episode_count,
        "source_samples": source_samples,
        "output_samples": output_samples,
        "pose_alignment_rejected": pose_alignment_rejected,
        "raw_source_root": (
            str(args.raw_source_root) if args.raw_source_root is not None else None
        ),
        "current_hz": args.current_hz,
        "history_hz": args.history_hz,
        "history_frames": args.history_frames,
        "trajectory_hz": args.trajectory_hz,
        "trajectory_horizon_seconds": 8.0 / args.trajectory_hz,
        "forward_speed_scale_mps": args.forward_speed_scale,
        "lateral_speed_scale_mps": args.lateral_speed_scale,
        "angular_speed_scale_radps": args.angular_speed_scale,
        "clip_normalized_speed": args.clip_normalized_speed,
        "max_alignment_error_ms": args.max_alignment_error_ms,
        "max_pose_alignment_error_ms": args.max_pose_alignment_error_ms,
        "archive_mode": args.archive_mode,
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[resample] wrote {output_path} ({output_samples} samples)")


if __name__ == "__main__":
    main()
