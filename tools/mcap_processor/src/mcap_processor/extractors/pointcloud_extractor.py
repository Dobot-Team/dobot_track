"""Extract point cloud messages."""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..core import BagReader
from ..utils import PathLike, ensure_dir, topic_to_filename
from .base import BaseExtractor


class PointcloudExtractor(BaseExtractor):
    """Extract Livox/custom pointcloud messages to CSV or NumPy files."""

    def __init__(
        self,
        reader: BagReader,
        output_dir: PathLike,
        topics: Optional[Sequence[str]] = None,
    ):
        super().__init__(reader, output_dir, topics)

    def _extract_points(self, msg) -> Optional[np.ndarray]:
        """Try to extract point coordinates from a pointcloud message."""
        ros_msg = msg.ros_msg
        # LivoxPointcloud custom message: points is a list of point objects
        if hasattr(ros_msg, "points") and isinstance(ros_msg.points, list):
            if not ros_msg.points:
                return None
            first = ros_msg.points[0]
            if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z"):
                pts = []
                for p in ros_msg.points:
                    pts.append([float(p.x), float(p.y), float(p.z)])
                return np.asarray(pts, dtype=np.float32)
        # Fallback: direct x/y/z arrays
        if hasattr(ros_msg, "x") and hasattr(ros_msg, "y") and hasattr(ros_msg, "z"):
            x = np.asarray(ros_msg.x, dtype=np.float32)
            y = np.asarray(ros_msg.y, dtype=np.float32)
            z = np.asarray(ros_msg.z, dtype=np.float32)
            return np.stack([x, y, z], axis=-1)
        return None

    def extract(self, max_messages: Optional[int] = None) -> None:
        """Extract point clouds to CSV and NumPy files."""
        ensure_dir(self.output_dir)

        counts: dict[str, int] = {}
        extracted = 0
        for msg in self.reader.iter_messages(topics=self.topics):
            points = self._extract_points(msg)
            if points is None:
                continue

            if max_messages is not None and extracted >= max_messages:
                break

            counts[msg.topic] = counts.get(msg.topic, 0) + 1
            extracted += 1
            base_name = topic_to_filename(msg.topic)
            timestamp_ns = msg.log_time_ns
            csv_path = self.output_dir / f"{base_name}_{timestamp_ns}.csv"
            np_path = self.output_dir / f"{base_name}_{timestamp_ns}.npy"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerows(points.tolist())

            np.save(np_path, points)

        print(f"Extracted point clouds to {self.output_dir}")
        for topic, count in sorted(counts.items()):
            print(f"  {topic}: {count}")
        if not counts:
            print("  (no pointcloud messages found)")
