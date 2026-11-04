"""Utility functions for MCAP processing."""

from __future__ import annotations

import bisect
import os
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union


PathLike = Union[str, os.PathLike]


def ns_to_seconds(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1e9


def ns_to_datetime(ns: int) -> datetime:
    """Convert ROS nanoseconds timestamp to UTC datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def ros_time_to_str(ns: int, fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> str:
    """Convert ROS nanoseconds timestamp to formatted string."""
    return ns_to_datetime(ns).strftime(fmt)


def ensure_dir(path: PathLike) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_mcap_files(bag_dir: PathLike) -> list[Path]:
    """Find all .mcap files in a rosbag2 directory, sorted by name."""
    bag_path = Path(bag_dir)
    if bag_path.is_file() and bag_path.suffix == ".mcap":
        return [bag_path]
    mcaps = sorted(bag_path.glob("*.mcap"))
    if not mcaps:
        raise FileNotFoundError(f"No .mcap files found in {bag_path}")
    return mcaps


def topic_to_filename(topic: str) -> str:
    """Convert a ROS topic name to a safe filename/directory name."""
    return topic.strip("/").replace("/", "_")


def ros_msg_to_dict(obj: Any) -> Any:
    """
    Recursively convert a decoded ROS message object into a JSON-serializable dict.
    Handles dataclasses, namedtuple-like ROS message classes, lists and primitives.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return list(obj)
    if isinstance(obj, (list, tuple)):
        return [ros_msg_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: ros_msg_to_dict(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: ros_msg_to_dict(getattr(obj, f.name)) for f in fields(obj)}

    # Fallback for dynamic mcap_ros2 message classes
    result: dict[str, Any] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        result[attr] = ros_msg_to_dict(value)
    return result


def find_nearest_timestamp(
    target_ns: int,
    timestamps: Sequence[int],
) -> Tuple[int, int]:
    """
    Find index and value of the timestamp closest to target_ns.
    Returns (index, timestamp_ns). Assumes timestamps is sorted ascending.
    """
    if not timestamps:
        raise ValueError("timestamps list is empty")
    idx = bisect.bisect_left(timestamps, target_ns)
    if idx == 0:
        return 0, timestamps[0]
    if idx == len(timestamps):
        return len(timestamps) - 1, timestamps[-1]
    before = timestamps[idx - 1]
    after = timestamps[idx]
    if abs(target_ns - before) <= abs(target_ns - after):
        return idx - 1, before
    return idx, after


def find_nearest_within_window(
    target_ns: int,
    entries: Sequence[Tuple[int, Any]],
    window_ns: int,
) -> Tuple[int, Any] | None:
    """
    Find the entry whose timestamp is closest to target_ns and within window_ns.
    entries is a list of (timestamp_ns, data). Returns (timestamp_ns, data) or None.
    """
    if not entries:
        return None
    timestamps = [e[0] for e in entries]
    idx, nearest_ts = find_nearest_timestamp(target_ns, timestamps)
    if abs(nearest_ts - target_ns) > window_ns:
        return None
    return entries[idx]
