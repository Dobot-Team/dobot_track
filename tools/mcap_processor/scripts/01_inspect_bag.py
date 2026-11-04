#!/usr/bin/env python3
"""Inspect a ROS2 MCAP bag and print topic statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcap_processor import BagAnalyzer, BagReader


DEFAULT_BAG = "/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a ROS2 MCAP bag")
    parser.add_argument(
        "--bag-dir",
        type=str,
        default=DEFAULT_BAG,
        help="Path to rosbag2 directory or .mcap file",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Limit number of messages for quick sampling",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON",
    )
    args = parser.parse_args()

    reader = BagReader(args.bag_dir)
    analyzer = BagAnalyzer(reader)
    summary = analyzer.summarize(max_messages=args.max_messages)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Bag: {summary.bag_dir}")
        print(f"Total messages: {summary.total_messages}")
        print()
        print(f"{'Topic':<50} {'Type':<45} {'Count':>8} {'Duration(s)':>12} {'Freq(Hz)':>10}")
        print("-" * 130)
        for topic, stats in sorted(summary.topics.items()):
            print(
                f"{topic:<50} {stats.msg_type:<45} {stats.count:>8} "
                f"{stats.duration_s:>12.3f} {stats.frequency_hz:>10.2f}"
            )


if __name__ == "__main__":
    main()
