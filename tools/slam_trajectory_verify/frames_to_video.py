#!/usr/bin/env python3
"""Combine rendered frames into an MP4 video."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def natural_sort_key(path: Path) -> list:
    """Sort paths with embedded numbers numerically."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float = 20.0,
    codec: str = "mp4v",
) -> None:
    """Read PNG frames and write an MP4 video."""
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("*.png"), key=natural_sort_key)
    if not frame_paths:
        raise ValueError(f"No PNG frames found in {frames_dir}")

    # Read first frame to get size
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    print(f"Writing {len(frame_paths)} frames to {output_path} @ {fps} fps")
    for i, frame_path in enumerate(frame_paths, 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: could not read {frame_path}, skipping")
            continue
        writer.write(frame)
        if i % 100 == 0:
            print(f"  written {i}/{len(frame_paths)}")

    writer.release()
    print(f"Done. Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine rendered frames into an MP4 video")
    parser.add_argument("--frames-dir", type=str, default="outputs/slam_trajectory_verify/frames")
    parser.add_argument("--output", type=str, default="outputs/slam_trajectory_verify/trajectory_video.mp4")
    parser.add_argument("--fps", type=float, default=20.0, help="Frames per second")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (default: mp4v)")
    args = parser.parse_args()

    frames_to_video(args.frames_dir, args.output, args.fps, args.codec)


if __name__ == "__main__":
    main()
