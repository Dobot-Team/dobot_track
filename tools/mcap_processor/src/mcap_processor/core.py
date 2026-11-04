"""Core MCAP reader wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

from mcap_ros2.reader import read_ros2_messages

from .utils import PathLike, find_mcap_files, ns_to_seconds


@dataclass
class DecodedMessage:
    """Decoded ROS2 message with metadata."""

    topic: str
    msg_type: str
    ros_msg: object
    log_time_ns: int
    publish_time_ns: int
    sequence: int


class BagReader:
    """High-level reader for ROS2 MCAP bags."""

    def __init__(self, bag_dir: PathLike):
        """
        Args:
            bag_dir: Path to a rosbag2 directory or a single .mcap file.
        """
        self.bag_dir = Path(bag_dir)
        self.mcap_files = find_mcap_files(self.bag_dir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.bag_dir}, files={len(self.mcap_files)})"

    def iter_messages(
        self,
        topics: Optional[Sequence[str]] = None,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
    ) -> Iterator[DecodedMessage]:
        """
        Iterate decoded messages across all mcap files.

        Args:
            topics: Optional list of topic names to filter.
            start_time_ns: Optional start time in nanoseconds.
            end_time_ns: Optional end time in nanoseconds.
        """
        topic_set = set(topics) if topics else None

        for mcap_file in self.mcap_files:
            for mcap_msg in read_ros2_messages(
                mcap_file,
                topics=topics,
                start_time=start_time_ns,
                end_time=end_time_ns,
            ):
                channel = mcap_msg.channel
                if topic_set and channel.topic not in topic_set:
                    continue
                yield DecodedMessage(
                    topic=channel.topic,
                    msg_type=mcap_msg.schema.name if mcap_msg.schema else "unknown",
                    ros_msg=mcap_msg.ros_msg,
                    log_time_ns=mcap_msg.log_time_ns,
                    publish_time_ns=mcap_msg.publish_time_ns,
                    sequence=getattr(mcap_msg, "sequence_count", 0),
                )

    def read_all(
        self,
        topics: Optional[Sequence[str]] = None,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
        max_messages: Optional[int] = None,
    ) -> List[DecodedMessage]:
        """Read all matching messages into a list."""
        messages = []
        for idx, msg in enumerate(self.iter_messages(topics, start_time_ns, end_time_ns)):
            if max_messages is not None and idx >= max_messages:
                break
            messages.append(msg)
        return messages
