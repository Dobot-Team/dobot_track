#!/usr/bin/env python3
"""
Time-synchronize VLA training samples using a camera topic as reference.

This script reuses data produced by the standalone extractors when available:
  - images     -> output/images/<topic>/<timestamp>.png
  - pointcloud -> output/pointclouds/<topic>_<timestamp>.npy
  - imu        -> output/imu/<topic>.csv

Only /lower/state and /lower/cmd are cached locally in _raw/cache.pkl because
there are no standalone extractors for those custom topics yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcap_processor import BagReader
from mcap_processor.utils import (
    ensure_dir,
    find_nearest_within_window,
    ns_to_seconds,
    topic_to_filename,
)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = str(PROJECT_ROOT / "output" / "synced_vla_samples")

# Shared raw output directories used by standalone extractors
DEFAULT_IMAGE_OUTPUT = str(PROJECT_ROOT / "output" / "images")
DEFAULT_POINTCLOUD_OUTPUT = str(PROJECT_ROOT / "output" / "pointclouds")
DEFAULT_IMU_OUTPUT = str(PROJECT_ROOT / "output" / "imu")

DEFAULT_SYNC_CONFIG = {
    "ref_topic": "/camera/camera/color/image_raw",
    "extra_camera_topics": [
        {"topic": "/camera/camera/infra1/image_rect_raw", "window_ms": 30},
        {"topic": "/camera/camera/infra2/image_rect_raw", "window_ms": 35},
        {"topic": "/camera/camera0/image_compressed", "window_ms": 15},
        {"topic": "/camera/camera2/image_compressed", "window_ms": 40},
    ],
    "sensor_topics": [
        {"topic": "/livox_Lidar_node/xyz/pointcloud", "window_ms": 100, "required": True},
        {"topic": "/camera/camera/imu", "window_ms": 20, "required": False},
        {"topic": "/livox_Lidar_node/imu/raw_data", "window_ms": 20, "required": False},
        {"topic": "/lower/state", "window_ms": 10, "required": False},
        {"topic": "/lower/cmd", "window_ms": 10, "required": False},
    ],
}


def load_sync_config(config_path: Optional[Path]) -> dict:
    """Load sync configuration from YAML or use defaults."""
    config = DEFAULT_SYNC_CONFIG.copy()
    if config_path is None:
        return config

    with open(config_path) as f:
        user_config = yaml.safe_load(f)

    if not user_config:
        return config

    # Merge user config with defaults
    if "ref_topic" in user_config:
        config["ref_topic"] = user_config["ref_topic"]
    if "extra_camera_topics" in user_config:
        config["extra_camera_topics"] = user_config["extra_camera_topics"]
    if "sensor_topics" in user_config:
        config["sensor_topics"] = user_config["sensor_topics"]

    return config


def get_ref_topic(config: dict) -> str:
    return config["ref_topic"]


def get_extra_camera_topics(config: dict) -> list[str]:
    return [item["topic"] for item in config["extra_camera_topics"]]


def get_extra_camera_windows(config: dict) -> dict[str, int]:
    return {item["topic"]: item["window_ms"] for item in config["extra_camera_topics"]}


def get_sensor_topics(config: dict) -> dict[str, int]:
    return {item["topic"]: item["window_ms"] for item in config["sensor_topics"]}


def get_required_sensors(config: dict) -> set[str]:
    return {
        item["topic"]
        for item in config["sensor_topics"]
        if item.get("required", False)
    }


# Initialize globals from defaults; main() may override via --config.
REF_TOPIC = DEFAULT_SYNC_CONFIG["ref_topic"]
EXTRA_CAMERA_TOPICS = get_extra_camera_topics(DEFAULT_SYNC_CONFIG)
ALL_CAMERA_TOPICS = [REF_TOPIC] + EXTRA_CAMERA_TOPICS
SYNC_TOPICS = get_sensor_topics(DEFAULT_SYNC_CONFIG)
EXTRA_CAMERA_WINDOWS_MS = get_extra_camera_windows(DEFAULT_SYNC_CONFIG)
REQUIRED_SENSORS = get_required_sensors(DEFAULT_SYNC_CONFIG)


def _image_ext_from_encoding(encoding: str) -> str:
    encoding = encoding.lower()
    if encoding in ("jpeg", "jpg", "rgb8_jpeg", "bgr8_jpeg"):
        return ".jpg"
    return ".png"


def convert_image_to_bgr(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to BGR numpy array for cv2.imwrite."""
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    if encoding in ("rgb8", "rgba8"):
        channels = 4 if "a" in encoding else 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if encoding in ("bgr8", "bgra8"):
        channels = 4 if "a" in encoding else 3
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))

    if encoding in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))

    if encoding in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width))

    # Fallback
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape((height, msg.step))


def save_camera_message(
    msg, topic: str, ts: int, image_output_dir: Path
) -> Optional[Path]:
    """Save a camera (Image or CompressedImage) message to shared image dir."""
    topic_dir = ensure_dir(image_output_dir / topic_to_filename(topic))
    msg_type = msg.msg_type.split("/")[-1]
    ros_msg = msg.ros_msg

    if msg_type == "CompressedImage":
        fmt = ros_msg.format.lower()
        ext = ".jpg" if "jpeg" in fmt or "jpg" in fmt else ".png"
        img_path = topic_dir / f"{ts}{ext}"
        with open(img_path, "wb") as f:
            f.write(bytes(ros_msg.data))
        return img_path

    if msg_type == "Image":
        try:
            img = convert_image_to_bgr(ros_msg)
            ext = _image_ext_from_encoding(ros_msg.encoding)
            img_path = topic_dir / f"{ts}{ext}"
            cv2.imwrite(str(img_path), img)
            return img_path
        except Exception as e:
            print(f"  Failed to convert {topic} @ {ts}: {e}")
            return None

    return None


def extract_pointcloud_points(msg) -> Optional[np.ndarray]:
    """Extract xyz points from LivoxPointcloud or similar custom messages."""
    ros_msg = msg
    if hasattr(ros_msg, "points") and isinstance(ros_msg.points, list):
        if not ros_msg.points:
            return None
        first = ros_msg.points[0]
        if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z"):
            pts = [[float(p.x), float(p.y), float(p.z)] for p in ros_msg.points]
            return np.asarray(pts, dtype=np.float32)
    if hasattr(ros_msg, "x") and hasattr(ros_msg, "y") and hasattr(ros_msg, "z"):
        x = np.asarray(ros_msg.x, dtype=np.float32)
        y = np.asarray(ros_msg.y, dtype=np.float32)
        z = np.asarray(ros_msg.z, dtype=np.float32)
        return np.stack([x, y, z], axis=-1)
    return None


def extract_imu_summary(msg) -> dict:
    """Fast extraction of IMU fields into CSV-compatible dict."""
    o = msg.orientation
    av = msg.angular_velocity
    la = msg.linear_acceleration
    return {
        "log_time_ns": None,  # filled by caller
        "orientation_x": o.x,
        "orientation_y": o.y,
        "orientation_z": o.z,
        "orientation_w": o.w,
        "angular_velocity_x": av.x,
        "angular_velocity_y": av.y,
        "angular_velocity_z": av.z,
        "linear_acceleration_x": la.x,
        "linear_acceleration_y": la.y,
        "linear_acceleration_z": la.z,
    }


def extract_lower_state_summary(msg) -> dict:
    """Fast extraction of LowerState fields relevant for VLA."""
    imu = msg.imu_state
    motors = []
    for m in msg.motor_state:
        motors.append(
            {
                "mode": m.mode,
                "q": m.q,
                "dq": m.dq,
                "tau_est": m.tau_est,
            }
        )
    return {
        "imu": {
            "quaternion": list(imu.quaternion),
            "gyroscope": list(imu.gyroscope),
            "accelerometer": list(imu.accelerometer),
            "rpy": list(imu.rpy),
        },
        "motors": motors,
    }


def extract_lower_cmd_summary(msg) -> dict:
    """Fast extraction of LowerCmd fields relevant for VLA."""
    motors = []
    for m in msg.motor_cmd:
        motors.append(
            {
                "mode": m.mode,
                "q": m.q,
                "dq": m.dq,
                "tau": m.tau,
                "kp": m.kp,
                "kd": m.kd,
            }
        )
    return {"motors": motors}


def _get_mcap_mtimes(reader: BagReader) -> dict[str, float]:
    """Return modification times for all mcap files."""
    return {str(p): p.stat().st_mtime for p in reader.mcap_files}


def _cache_path(output_dir: Path) -> Path:
    """Local cache for lower_state / lower_cmd (no standalone extractor yet)."""
    return output_dir / "_raw" / "cache.pkl"


def _save_state_cmd_cache(
    output_dir: Path, reader: BagReader, lower_states: list, lower_cmds: list
) -> None:
    cache = {
        "version": 1,
        "bag_dir": str(reader.bag_dir),
        "mcap_mtimes": _get_mcap_mtimes(reader),
        "/lower/state": lower_states,
        "/lower/cmd": lower_cmds,
    }
    cache_path = _cache_path(output_dir)
    ensure_dir(cache_path.parent)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def _load_state_cmd_cache_if_valid(output_dir: Path, reader: BagReader) -> Optional[dict]:
    cache_path = _cache_path(output_dir)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("version") != 1:
            return None
        if cache.get("bag_dir") != str(reader.bag_dir):
            return None
        if cache.get("mcap_mtimes") != _get_mcap_mtimes(reader):
            return None
        return cache
    except Exception as e:
        print(f"  State/cmd cache invalid ({e}), will rebuild.")
        return None


def _scan_image_dir(image_output_dir: Path, topic: str) -> list[tuple[int, Path]]:
    """Scan extractor output dir for images of a specific topic."""
    topic_dir = image_output_dir / topic_to_filename(topic)
    entries = []
    if topic_dir.exists():
        for ext in (".png", ".jpg"):
            for p in topic_dir.glob(f"*{ext}"):
                try:
                    ts = int(p.stem)
                    entries.append((ts, p))
                except ValueError:
                    continue
    return sorted(entries, key=lambda x: x[0])


def _scan_pointcloud_dir(pc_output_dir: Path, topic: str) -> list[tuple[int, Path]]:
    """Scan extractor output dir for pointcloud files of a specific topic."""
    prefix = topic_to_filename(topic) + "_"
    entries = []
    if pc_output_dir.exists():
        for p in pc_output_dir.glob(f"{prefix}*.npy"):
            try:
                ts = int(p.stem[len(prefix) :])
                entries.append((ts, p))
            except ValueError:
                continue
    return sorted(entries, key=lambda x: x[0])


def _parse_imu_csv(csv_path: Path) -> list[tuple[int, dict]]:
    """Parse an IMU CSV back into (timestamp_ns, dict) records."""
    if not csv_path.exists():
        return []
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["log_time_ns"])
            records.append((ts, dict(row)))
    return records


def collect_messages(
    reader: BagReader,
    output_dir: Path,
    image_output_dir: Path,
    pc_output_dir: Path,
    imu_output_dir: Path,
    force_rebuild: bool = False,
):
    """
    Collect all messages needed for synchronization.
    Reuses outputs from standalone extractors when available.
    Only falls back to mcap extraction for missing data.
    """
    all_topics = ALL_CAMERA_TOPICS + ["/livox_Lidar_node/xyz/pointcloud"] + list(SYNC_TOPICS.keys())

    # 1) Try to reuse standalone extractor outputs
    camera_data: dict[str, list[tuple[int, Path]]] = {
        topic: _scan_image_dir(image_output_dir, topic) for topic in ALL_CAMERA_TOPICS
    }
    pointclouds = _scan_pointcloud_dir(pc_output_dir, "/livox_Lidar_node/xyz/pointcloud")
    imu_camera = _parse_imu_csv(imu_output_dir / f"{topic_to_filename('/camera/camera/imu')}.csv")
    imu_livox = _parse_imu_csv(
        imu_output_dir / f"{topic_to_filename('/livox_Lidar_node/imu/raw_data')}.csv"
    )

    # 2) Load or build lower_state / lower_cmd cache
    state_cmd_cache = None if force_rebuild else _load_state_cmd_cache_if_valid(output_dir, reader)
    if state_cmd_cache is not None:
        print("Using cached lower_state / lower_cmd data.")
        lower_states = state_cmd_cache.get("/lower/state", [])
        lower_cmds = state_cmd_cache.get("/lower/cmd", [])
    else:
        lower_states = []
        lower_cmds = []

    # 3) Determine what is still missing
    need_cameras = {topic: len(camera_data[topic]) == 0 for topic in ALL_CAMERA_TOPICS}
    need_pointclouds = len(pointclouds) == 0
    need_imu_camera = len(imu_camera) == 0
    need_imu_livox = len(imu_livox) == 0
    need_state_cmd = state_cmd_cache is None

    missing = []
    for topic, needed in need_cameras.items():
        if needed:
            missing.append(topic)
    if need_pointclouds:
        missing.append("/livox_Lidar_node/xyz/pointcloud")
    if need_imu_camera:
        missing.append("/camera/camera/imu")
    if need_imu_livox:
        missing.append("/livox_Lidar_node/imu/raw_data")
    if need_state_cmd:
        missing.append("/lower/state + /lower/cmd")

    if not missing:
        print("All required data already extracted. Skipping mcap read.")
        return {
            "ref_images": camera_data[REF_TOPIC],
            "extra_cameras": {t: camera_data[t] for t in EXTRA_CAMERA_TOPICS},
            "pointclouds": pointclouds,
            "/camera/camera/imu": imu_camera,
            "/livox_Lidar_node/imu/raw_data": imu_livox,
            "/lower/state": lower_states,
            "/lower/cmd": lower_cmds,
        }

    print(f"Missing data, will extract from mcap: {', '.join(missing)}")
    print("Reading bag and extracting missing messages...")

    for topic in ALL_CAMERA_TOPICS:
        if need_cameras[topic]:
            ensure_dir(image_output_dir / topic_to_filename(topic))
    ensure_dir(pc_output_dir)
    ensure_dir(imu_output_dir)

    imu_camera_buffer = [] if need_imu_camera else None
    imu_livox_buffer = [] if need_imu_livox else None

    total = 0
    for msg in reader.iter_messages(topics=all_topics):
        total += 1
        ts = msg.log_time_ns
        topic = msg.topic

        if topic in ALL_CAMERA_TOPICS:
            if not need_cameras[topic]:
                continue
            img_path = save_camera_message(msg, topic, ts, image_output_dir)
            if img_path is not None:
                camera_data[topic].append((ts, img_path))

        elif topic == "/livox_Lidar_node/xyz/pointcloud" and need_pointclouds:
            pts = extract_pointcloud_points(msg.ros_msg)
            if pts is not None:
                pc_path = pc_output_dir / f"{topic_to_filename(topic)}_{ts}.npy"
                np.save(pc_path, pts)
                pointclouds.append((ts, pc_path))

        elif topic == "/camera/camera/imu" and need_imu_camera:
            d = extract_imu_summary(msg.ros_msg)
            d["log_time_ns"] = ts
            imu_camera_buffer.append(d)

        elif topic == "/livox_Lidar_node/imu/raw_data" and need_imu_livox:
            d = extract_imu_summary(msg.ros_msg)
            d["log_time_ns"] = ts
            imu_livox_buffer.append(d)

        elif topic == "/lower/state" and need_state_cmd:
            lower_states.append((ts, extract_lower_state_summary(msg.ros_msg)))

        elif topic == "/lower/cmd" and need_state_cmd:
            lower_cmds.append((ts, extract_lower_cmd_summary(msg.ros_msg)))

        if total % 10000 == 0:
            print(f"  processed {total} messages...")

    print(f"Done. Total messages processed: {total}")

    # Save newly extracted IMU CSVs to shared output dir
    if need_imu_camera and imu_camera_buffer:
        csv_path = imu_output_dir / f"{topic_to_filename('/camera/camera/imu')}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=imu_camera_buffer[0].keys())
            writer.writeheader()
            writer.writerows(imu_camera_buffer)
        imu_camera = [(d["log_time_ns"], d) for d in imu_camera_buffer]
        print(f"  Saved {len(imu_camera_buffer)} camera IMU rows to {csv_path}")

    if need_imu_livox and imu_livox_buffer:
        csv_path = imu_output_dir / f"{topic_to_filename('/livox_Lidar_node/imu/raw_data')}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=imu_livox_buffer[0].keys())
            writer.writeheader()
            writer.writerows(imu_livox_buffer)
        imu_livox = [(d["log_time_ns"], d) for d in imu_livox_buffer]
        print(f"  Saved {len(imu_livox_buffer)} Livox IMU rows to {csv_path}")

    if need_state_cmd:
        lower_states = sorted(lower_states, key=lambda x: x[0])
        lower_cmds = sorted(lower_cmds, key=lambda x: x[0])
        _save_state_cmd_cache(output_dir, reader, lower_states, lower_cmds)

    return {
        "ref_images": sorted(camera_data[REF_TOPIC], key=lambda x: x[0]),
        "extra_cameras": {
            topic: sorted(camera_data[topic], key=lambda x: x[0]) for topic in EXTRA_CAMERA_TOPICS
        },
        "pointclouds": sorted(pointclouds, key=lambda x: x[0]),
        "/camera/camera/imu": sorted(imu_camera, key=lambda x: x[0]),
        "/livox_Lidar_node/imu/raw_data": sorted(imu_livox, key=lambda x: x[0]),
        "/lower/state": lower_states,
        "/lower/cmd": lower_cmds,
    }


def build_synced_samples(
    data: dict,
    output_dir: Path,
    max_samples: Optional[int] = None,
    required_sensors: Optional[set[str]] = None,
) -> list[dict]:
    if required_sensors is None:
        required_sensors = REQUIRED_SENSORS
    """Build synced samples centered on each reference camera frame."""
    samples: list[dict] = []
    ref_images = data["ref_images"]
    extra_cameras = data["extra_cameras"]
    pointclouds = data["pointclouds"]
    imu_camera = data["/camera/camera/imu"]
    imu_livox = data["/livox_Lidar_node/imu/raw_data"]
    lower_states = data["/lower/state"]
    lower_cmds = data["/lower/cmd"]

    sync_sources = {
        "/livox_Lidar_node/xyz/pointcloud": pointclouds,
        "/camera/camera/imu": imu_camera,
        "/livox_Lidar_node/imu/raw_data": imu_livox,
        "/lower/state": lower_states,
        "/lower/cmd": lower_cmds,
    }

    total_ref = len(ref_images)
    for idx, (ref_ts, img_path) in enumerate(ref_images):
        if max_samples is not None and len(samples) >= max_samples:
            break

        sample_dir = ensure_dir(output_dir / f"sample_{ref_ts}")

        # Symlink reference image into sample dir
        sample_img = sample_dir / "image.png"
        if not sample_img.exists():
            sample_img.symlink_to(Path(img_path).resolve())

        synced: dict[str, Any] = {
            "ref_topic": REF_TOPIC,
            "ref_timestamp_ns": ref_ts,
            "ref_timestamp_iso": None,
            "image": str(sample_img.relative_to(output_dir)),
            "cameras": {},
        }

        # Symlink extra camera frames
        extra_ok = True
        for topic in EXTRA_CAMERA_TOPICS:
            window_ns = EXTRA_CAMERA_WINDOWS_MS[topic] * 1_000_000
            result = find_nearest_within_window(ref_ts, extra_cameras[topic], window_ns)
            if result is None:
                synced["cameras"][topic] = None
                extra_ok = False
                continue
            nearest_ts, src_path = result
            safe_name = topic_to_filename(topic)
            ext = src_path.suffix
            sample_cam = sample_dir / f"{safe_name}{ext}"
            if not sample_cam.exists():
                sample_cam.symlink_to(Path(src_path).resolve())
            synced["cameras"][topic] = {
                "timestamp_ns": nearest_ts,
                "offset_ms": round(ns_to_seconds(nearest_ts - ref_ts) * 1000.0, 3),
                "path": str(sample_cam.relative_to(output_dir)),
            }

        if not extra_ok:
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue

        # Pointcloud and other sensors
        synced_ok = True
        for topic, window_ms in SYNC_TOPICS.items():
            window_ns = window_ms * 1_000_000
            result = find_nearest_within_window(ref_ts, sync_sources[topic], window_ns)
            if result is None:
                synced[topic] = None
                if topic in required_sensors:
                    synced_ok = False
            else:
                nearest_ts, nearest_data = result
                offset_ms = ns_to_seconds(nearest_ts - ref_ts) * 1000.0
                if topic == "/livox_Lidar_node/xyz/pointcloud":
                    src_pc = Path(nearest_data).resolve()
                    sample_pc = sample_dir / "pointcloud.npy"
                    if not sample_pc.exists():
                        sample_pc.symlink_to(src_pc)
                    synced[topic] = {
                        "timestamp_ns": nearest_ts,
                        "offset_ms": round(offset_ms, 3),
                        "path": str(sample_pc.relative_to(output_dir)),
                        "num_points": int(np.load(src_pc).shape[0]),
                    }
                else:
                    safe_name = topic_to_filename(topic)
                    json_path = sample_dir / f"{safe_name}.json"
                    with open(json_path, "w") as f:
                        json.dump(nearest_data, f, indent=2, ensure_ascii=False)
                    synced[topic] = {
                        "timestamp_ns": nearest_ts,
                        "offset_ms": round(offset_ms, 3),
                        "path": str(json_path.relative_to(output_dir)),
                    }

        if not synced_ok:
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue

        synced["ref_timestamp_iso"] = _ns_to_iso(ref_ts)
        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(synced, f, indent=2, ensure_ascii=False)

        samples.append(synced)

        if (idx + 1) % 50 == 0 or idx == total_ref - 1:
            print(f"  synced {len(samples)}/{idx + 1} reference frames...")

    return samples


def _ns_to_iso(ns: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def process_single_bag(
    bag_dir: Path,
    output_dir: Path,
    image_output_dir: Path,
    pc_output_dir: Path,
    imu_output_dir: Path,
    max_samples: Optional[int],
    force_rebuild: bool,
) -> None:
    """Run the full sync pipeline for one bag directory."""
    ensure_dir(output_dir)
    ensure_dir(image_output_dir)
    ensure_dir(pc_output_dir)
    ensure_dir(imu_output_dir)

    reader = BagReader(bag_dir)
    data = collect_messages(
        reader,
        output_dir,
        image_output_dir=image_output_dir,
        pc_output_dir=pc_output_dir,
        imu_output_dir=imu_output_dir,
        force_rebuild=force_rebuild,
    )

    print("\nReference images:", len(data["ref_images"]))
    for topic, entries in data["extra_cameras"].items():
        print(f"Extra camera {topic}: {len(entries)}")
    print("Pointclouds:", len(data["pointclouds"]))
    for topic in SYNC_TOPICS:
        if topic == "/livox_Lidar_node/xyz/pointcloud":
            continue
        print(f"{topic}: {len(data[topic])}")

    print("\nBuilding synced samples...")
    samples = build_synced_samples(
        data, output_dir, max_samples=max_samples, required_sensors=REQUIRED_SENSORS
    )

    # Save index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "ref_topic": REF_TOPIC,
                "extra_camera_topics": EXTRA_CAMERA_TOPICS,
                "sync_topics": SYNC_TOPICS,
                "extra_camera_windows_ms": EXTRA_CAMERA_WINDOWS_MS,
                "num_samples": len(samples),
                "samples": samples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDone. Generated {len(samples)} synced samples in {output_dir}")
    print(f"Index saved to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize VLA samples around a camera topic"
    )
    parser.add_argument(
        "--bag-dir",
        type=str,
        default=None,
        help="Path to a single rosbag2 directory",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Path to a text file with one rosbag2 directory per line",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory for synced samples (single bag) or parent directory (batch)",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Directory for extracted images (single bag) or parent directory (batch)",
    )
    parser.add_argument(
        "--pointcloud-output-dir",
        type=str,
        default=DEFAULT_POINTCLOUD_OUTPUT,
        help="Directory for extracted pointclouds (single bag) or parent directory (batch)",
    )
    parser.add_argument(
        "--imu-output-dir",
        type=str,
        default=DEFAULT_IMU_OUTPUT,
        help="Directory for extracted IMU CSVs (single bag) or parent directory (batch)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of synced samples to generate per bag",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cache and re-read the mcap from scratch",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML sync configuration file",
    )
    args = parser.parse_args()

    if not args.bag_dir and not args.batch_file:
        parser.error("Either --bag-dir or --batch-file must be provided")
    if args.bag_dir and args.batch_file:
        parser.error("Cannot use both --bag-dir and --batch-file")

    # Load config and update module-level topic globals
    global REF_TOPIC, EXTRA_CAMERA_TOPICS, ALL_CAMERA_TOPICS, SYNC_TOPICS
    global EXTRA_CAMERA_WINDOWS_MS, REQUIRED_SENSORS
    config = load_sync_config(Path(args.config) if args.config else None)
    REF_TOPIC = get_ref_topic(config)
    EXTRA_CAMERA_TOPICS = get_extra_camera_topics(config)
    ALL_CAMERA_TOPICS = [REF_TOPIC] + EXTRA_CAMERA_TOPICS
    SYNC_TOPICS = get_sensor_topics(config)
    EXTRA_CAMERA_WINDOWS_MS = get_extra_camera_windows(config)
    REQUIRED_SENSORS = get_required_sensors(config)

    # Determine list of bags to process
    if args.batch_file:
        with open(args.batch_file) as f:
            bag_dirs = [line.strip() for line in f if line.strip()]
    else:
        bag_dirs = [args.bag_dir]

    base_output_dir = Path(args.output_dir)
    base_image_dir = Path(args.image_output_dir)
    base_pc_dir = Path(args.pointcloud_output_dir)
    base_imu_dir = Path(args.imu_output_dir)

    for bag_dir_str in bag_dirs:
        bag_dir = Path(bag_dir_str)
        if not bag_dir.exists():
            print(f"\nSkipping non-existent bag: {bag_dir}")
            continue

        bag_name = bag_dir.name
        if len(bag_dirs) == 1:
            # Single bag mode: use output dirs directly
            output_dir = base_output_dir
            image_output_dir = base_image_dir
            pc_output_dir = base_pc_dir
            imu_output_dir = base_imu_dir
        else:
            # Batch mode: create per-bag subdirectories
            output_dir = base_output_dir / bag_name
            image_output_dir = base_image_dir / bag_name
            pc_output_dir = base_pc_dir / bag_name
            imu_output_dir = base_imu_dir / bag_name

        print("=" * 70)
        print(f"Processing bag: {bag_dir}")
        print(f"Output: {output_dir}")
        print("=" * 70)

        process_single_bag(
            bag_dir=bag_dir,
            output_dir=output_dir,
            image_output_dir=image_output_dir,
            pc_output_dir=pc_output_dir,
            imu_output_dir=imu_output_dir,
            max_samples=args.max_samples,
            force_rebuild=args.force_rebuild,
        )


if __name__ == "__main__":
    main()
