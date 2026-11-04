"""Extract images from MCAP bag topics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..core import BagReader
from ..utils import PathLike, ensure_dir, topic_to_filename
from .base import BaseExtractor


SUPPORTED_IMAGE_TYPES = {"Image", "CompressedImage"}


def _get_image_format(encoding: str) -> str:
    """Infer file extension from image encoding."""
    encoding = encoding.lower()
    if encoding in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1", "8uc3"):
        return ".png"
    if encoding in ("jpeg", "jpg", "rgb8_jpeg", "bgr8_jpeg"):
        return ".jpg"
    if encoding in ("png",):
        return ".png"
    return ".png"


def _convert_image(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to numpy array."""
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    if encoding in ("rgb8", "rgba8"):
        channels = 4 if "a" in encoding else 3
        dtype = np.uint8
        img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width, channels))
        # OpenCV uses BGR by default; convert RGB to BGR for saving
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if encoding in ("bgr8", "bgra8"):
        channels = 4 if "a" in encoding else 3
        dtype = np.uint8
        return np.frombuffer(msg.data, dtype=dtype).reshape((height, width, channels))

    if encoding in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))

    if encoding in ("16uc1", "mono16"):
        # depth image typically 16UC1
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width))

    if encoding in ("32fc1",):
        return np.frombuffer(msg.data, dtype=np.float32).reshape((height, width))

    # Fallback: try to interpret from step size
    bytes_per_row = msg.step
    total_bytes = bytes_per_row * height
    arr = np.frombuffer(msg.data[:total_bytes], dtype=np.uint8)
    return arr.reshape((height, bytes_per_row))


class ImageExtractor(BaseExtractor):
    """Extract sensor_msgs/Image and sensor_msgs/CompressedImage."""

    def __init__(
        self,
        reader: BagReader,
        output_dir: PathLike,
        topics: Optional[Sequence[str]] = None,
        save_raw_compressed: bool = True,
    ):
        super().__init__(reader, output_dir, topics)
        self.save_raw_compressed = save_raw_compressed

    def extract(self, max_messages: Optional[int] = None) -> None:
        """Extract images to output_dir/<topic>/timestamp.ext."""
        ensure_dir(self.output_dir)
        counts: dict[str, int] = {}
        extracted = 0

        for msg in self.reader.iter_messages(topics=self.topics):
            msg_type = msg.msg_type.split("/")[-1]
            if msg_type not in SUPPORTED_IMAGE_TYPES:
                continue

            if max_messages is not None and extracted >= max_messages:
                break

            timestamp_ns = msg.log_time_ns
            topic_dir = ensure_dir(self.output_dir / topic_to_filename(msg.topic))

            if msg_type == "CompressedImage":
                fmt = msg.ros_msg.format.lower()
                ext = ".jpg" if "jpeg" in fmt or "jpg" in fmt else ".png"
                filename = topic_dir / f"{timestamp_ns}{ext}"
                data = bytes(msg.ros_msg.data)
                with open(filename, "wb") as f:
                    f.write(data)
                if self.save_raw_compressed:
                    raw_filename = topic_dir / f"{timestamp_ns}.bin"
                    with open(raw_filename, "wb") as f:
                        f.write(data)
                counts[msg.topic] = counts.get(msg.topic, 0) + 1
                extracted += 1
            elif msg_type == "Image":
                ext = _get_image_format(msg.ros_msg.encoding)
                filename = topic_dir / f"{timestamp_ns}{ext}"
                try:
                    img = _convert_image(msg.ros_msg)
                    cv2.imwrite(str(filename), img)
                    counts[msg.topic] = counts.get(msg.topic, 0) + 1
                    extracted += 1
                except Exception as e:
                    print(f"Failed to convert {msg.topic} @ {timestamp_ns}: {e}")

        print(f"Extracted images to {self.output_dir}")
        for topic, count in sorted(counts.items()):
            print(f"  {topic}: {count}")
        if not counts:
            print("  (no image messages found)")
