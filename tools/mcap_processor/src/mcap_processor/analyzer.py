"""Bag analysis utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .core import BagReader
from .utils import ns_to_datetime, ns_to_seconds, topic_to_filename


@dataclass
class TopicStats:
    """Statistics for a single topic."""

    topic: str
    msg_type: str = "unknown"
    count: int = 0
    first_log_time_ns: Optional[int] = None
    last_log_time_ns: Optional[int] = None

    @property
    def duration_s(self) -> float:
        if self.first_log_time_ns is None or self.last_log_time_ns is None:
            return 0.0
        return ns_to_seconds(self.last_log_time_ns - self.first_log_time_ns)

    @property
    def frequency_hz(self) -> float:
        if self.count <= 1 or self.duration_s <= 0:
            return 0.0
        return (self.count - 1) / self.duration_s

    def update(self, log_time_ns: int, msg_type: str) -> None:
        self.count += 1
        self.msg_type = msg_type
        if self.first_log_time_ns is None:
            self.first_log_time_ns = log_time_ns
        self.last_log_time_ns = log_time_ns


@dataclass
class BagSummary:
    """Summary of a bag."""

    bag_dir: str
    total_messages: int = 0
    topics: Dict[str, TopicStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bag_dir": self.bag_dir,
            "total_messages": self.total_messages,
            "topics": {
                topic: {
                    "msg_type": stats.msg_type,
                    "count": stats.count,
                    "first_time": ns_to_datetime(stats.first_log_time_ns).isoformat()
                    if stats.first_log_time_ns
                    else None,
                    "last_time": ns_to_datetime(stats.last_log_time_ns).isoformat()
                    if stats.last_log_time_ns
                    else None,
                    "duration_s": round(stats.duration_s, 3),
                    "frequency_hz": round(stats.frequency_hz, 2),
                }
                for topic, stats in sorted(self.topics.items())
            },
        }


class BagAnalyzer:
    """Analyze MCAP bag contents."""

    def __init__(self, reader: BagReader):
        self.reader = reader

    def summarize(
        self,
        topics: Optional[Sequence[str]] = None,
        max_messages: Optional[int] = None,
    ) -> BagSummary:
        """
        Compute message counts, durations and frequencies per topic.

        Args:
            topics: Optional topic filter.
            max_messages: Optional limit for quick sampling.
        """
        summary = BagSummary(bag_dir=str(self.reader.bag_dir))

        for idx, msg in enumerate(self.reader.iter_messages(topics=topics)):
            if max_messages is not None and idx >= max_messages:
                break
            stats = summary.topics.setdefault(
                msg.topic, TopicStats(topic=msg.topic)
            )
            stats.update(msg.log_time_ns, msg.msg_type)
            summary.total_messages += 1

        return summary

    def get_topics(self) -> List[str]:
        """Return list of topics present in the bag."""
        topics = set()
        for msg in self.reader.iter_messages():
            topics.add(msg.topic)
        return sorted(topics)
