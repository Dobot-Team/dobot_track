"""Extract IMU messages to CSV/JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from ..core import BagReader
from ..utils import PathLike, ensure_dir, topic_to_filename
from .base import BaseExtractor


class ImuExtractor(BaseExtractor):
    """Extract sensor_msgs/Imu to CSV."""

    def __init__(
        self,
        reader: BagReader,
        output_dir: PathLike,
        topics: Optional[Sequence[str]] = None,
        output_format: str = "csv",
    ):
        super().__init__(reader, output_dir, topics)
        self.output_format = output_format.lower()
        if self.output_format not in ("csv", "json"):
            raise ValueError("output_format must be 'csv' or 'json'")

    def _msg_to_dict(self, msg) -> dict:
        imu = msg.ros_msg
        return {
            "log_time_ns": msg.log_time_ns,
            "orientation_x": imu.orientation.x,
            "orientation_y": imu.orientation.y,
            "orientation_z": imu.orientation.z,
            "orientation_w": imu.orientation.w,
            "angular_velocity_x": imu.angular_velocity.x,
            "angular_velocity_y": imu.angular_velocity.y,
            "angular_velocity_z": imu.angular_velocity.z,
            "linear_acceleration_x": imu.linear_acceleration.x,
            "linear_acceleration_y": imu.linear_acceleration.y,
            "linear_acceleration_z": imu.linear_acceleration.z,
        }

    def extract(self, max_messages: Optional[int] = None) -> None:
        """Extract IMU messages to one file per topic."""
        ensure_dir(self.output_dir)

        rows_by_topic: dict[str, list[dict]] = {}

        for idx, msg in enumerate(self.reader.iter_messages(topics=self.topics)):
            if max_messages is not None and idx >= max_messages:
                break
            if msg.msg_type.split("/")[-1] != "Imu":
                continue
            rows_by_topic.setdefault(msg.topic, []).append(self._msg_to_dict(msg))

        for topic, rows in rows_by_topic.items():
            base_name = topic_to_filename(topic)
            if self.output_format == "csv":
                path = self.output_dir / f"{base_name}.csv"
                if rows:
                    with open(path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"Wrote {len(rows)} IMU rows to {path}")
            else:
                path = self.output_dir / f"{base_name}.json"
                with open(path, "w") as f:
                    json.dump(rows, f, indent=2)
                print(f"Wrote {len(rows)} IMU rows to {path}")
