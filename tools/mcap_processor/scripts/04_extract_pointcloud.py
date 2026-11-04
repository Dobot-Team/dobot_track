#!/usr/bin/env python3
"""Extract pointcloud messages from a ROS2 MCAP bag."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcap_processor import BagReader
from mcap_processor.extractors import PointcloudExtractor


DEFAULT_BAG = "/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000"
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = str(PROJECT_ROOT / "output" / "pointclouds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pointclouds from a MCAP bag")
    parser.add_argument(
        "--bag-dir",
        type=str,
        default=DEFAULT_BAG,
        help="Path to rosbag2 directory or .mcap file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        default=None,
        help="Specific pointcloud topics to extract",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Maximum number of messages to extract",
    )
    args = parser.parse_args()

    reader = BagReader(args.bag_dir)
    extractor = PointcloudExtractor(
        reader=reader,
        output_dir=args.output_dir,
        topics=args.topics,
    )
    extractor.extract(max_messages=args.max_messages)


if __name__ == "__main__":
    main()
