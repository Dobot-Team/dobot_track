#!/usr/bin/env python3
"""Generate a minimal synthetic NuScenes-mini dataset for smoke testing.

This creates the required JSON tables and dummy JPEG images so that the
nuscenes-devkit can load the dataset and our VLA data module can iterate over
it without downloading the real 4 GB NuScenes-mini archive.

Usage:
    python tools/gen_dummy_nuscenes.py --output /home/jovyan/work/data/nuscenes-mini \
        --num_scenes 3 --frames_per_scene 30
"""
import argparse
import json
import os
import uuid
from typing import Dict, List

import numpy as np
from PIL import Image
from pyquaternion import Quaternion


def gen_token(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def write_json(path: str, data: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_dummy_image(path: str, size: tuple = (800, 450)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=85)


def generate_scene_data(
    scene_idx: int,
    n_frames: int,
    dataroot: str,
    sample_dt_us: int = 500_000,  # 0.5 s keyframes
    camera_dt_us: int = 100_000,  # 10 Hz camera
) -> Dict[str, List[dict]]:
    """Generate all records for one synthetic scene."""
    tables = {
        "sample": [],
        "sample_data": [],
        "sample_annotation": [],
        "ego_pose": [],
        "calibrated_sensor": [],
        "scene": [],
        "log": [],
    }

    log_token = gen_token("log_")
    scene_token = gen_token("scene_")
    tables["log"].append({
        "token": log_token,
        "logfile": f"scene_{scene_idx:03d}",
        "vehicle": "synthetic_vehicle",
        "date_captured": "2024-01-01",
        "location": "synthetic",
    })
    tables["scene"].append({
        "token": scene_token,
        "log_token": log_token,
        "nbr_samples": n_frames,
        "first_sample_token": "",  # filled later
        "last_sample_token": "",
        "name": f"scene-{scene_idx:04d}",
        "description": "Synthetic scene for VLA smoke test.",
    })

    # Synthetic ego motion: drive forward with slight sinusoidal lateral motion.
    start_time_us = 1_000_000_000 + scene_idx * 1_000_000_000
    keyframe_times = [start_time_us + i * sample_dt_us for i in range(n_frames)]
    camera_times = []
    t = start_time_us
    while t <= keyframe_times[-1] + camera_dt_us:
        camera_times.append(t)
        t += camera_dt_us

    # Create all camera frames (sample_data) and ego poses.
    sd_tokens = []
    for i, ct in enumerate(camera_times):
        sd_token = gen_token("sd_")
        sd_tokens.append(sd_token)
        x = 0.0 + (ct - start_time_us) * 1e-6 * 5.0  # 5 m/s forward
        y = np.sin((ct - start_time_us) * 1e-6 * 0.5) * 2.0
        z = 0.0
        yaw = 0.0
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        ep_token = gen_token("ep_")
        tables["ego_pose"].append({
            "token": ep_token,
            "timestamp": ct,
            "rotation": [q.w, q.x, q.y, q.z],
            "translation": [float(x), float(y), float(z)],
        })

        # Calibrated sensor (camera extrinsic/intrinsic) - reuse one per scene.
        cs_token = gen_token(f"cs_scene{scene_idx}_")
        tables["calibrated_sensor"].append({
            "token": cs_token,
            "sensor_token": "sensor_cam_front",
            "translation": [0.0, 0.0, 1.6],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [
                [800.0, 0.0, 400.0],
                [0.0, 800.0, 225.0],
                [0.0, 0.0, 1.0],
            ],
        })

        img_filename = f"samples/CAM_FRONT/scene{scene_idx:03d}_frame{i:04d}.jpg"
        create_dummy_image(os.path.join(dataroot, img_filename), size=(800, 450))

        tables["sample_data"].append({
            "token": sd_token,
            "sample_token": "",  # filled for keyframes
            "ego_pose_token": ep_token,
            "calibrated_sensor_token": cs_token,
            "filename": img_filename,
            "fileformat": "jpg",
            "width": 800,
            "height": 450,
            "timestamp": ct,
            "is_key_frame": ct in keyframe_times,
            "next": "",  # filled later
            "prev": "",  # filled later
        })

    # Link sample_data next/prev.
    for i, sd in enumerate(tables["sample_data"]):
        if i > 0:
            sd["prev"] = tables["sample_data"][i - 1]["token"]
        if i < len(tables["sample_data"]) - 1:
            sd["next"] = tables["sample_data"][i + 1]["token"]

    # Create keyframe samples and bind to camera sample_data.
    prev_sample_token = ""
    first_sample_token = ""
    for kf_idx, kf_time in enumerate(keyframe_times):
        sample_token = gen_token("sample_")
        if kf_idx == 0:
            first_sample_token = sample_token

        # Find sample_data at this exact keyframe time.
        cam_sd = next(sd for sd in tables["sample_data"] if sd["timestamp"] == kf_time)
        cam_sd["sample_token"] = sample_token

        tables["sample"].append({
            "token": sample_token,
            "timestamp": kf_time,
            "prev": prev_sample_token,
            "next": "",
            "scene_token": scene_token,
            "data": {
                "CAM_FRONT": cam_sd["token"],
            },
        })
        if prev_sample_token:
            tables["sample"][-2]["next"] = sample_token
        prev_sample_token = sample_token

    tables["scene"][0]["first_sample_token"] = first_sample_token
    tables["scene"][0]["last_sample_token"] = prev_sample_token

    return tables


def merge_tables(all_tables: List[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    merged = {}
    for key in all_tables[0].keys():
        merged[key] = []
        for tables in all_tables:
            merged[key].extend(tables[key])
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/home/jovyan/work/data/nuscenes-mini")
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument("--frames_per_scene", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Static tables (shared across scenes).
    static_tables = {
        "category": [
            {"token": gen_token("cat_"), "name": "vehicle.car", "description": "Car"},
        ],
        "attribute": [
            {"token": gen_token("attr_"), "name": "vehicle.moving", "description": "Moving"},
        ],
        "visibility": [
            {"token": gen_token("vis_"), "description": "v0-40", "level": "0"},
        ],
        "instance": [],
        "sensor": [
            {
                "token": "sensor_cam_front",
                "channel": "CAM_FRONT",
                "modality": "camera",
            }
        ],
        "map": [
            {"token": gen_token("map_"), "category": "semantic_prior", "filename": "maps/empty.png", "canvas_edge": 100, "log_tokens": []},
        ],
    }

    all_scene_tables = []
    for scene_idx in range(args.num_scenes):
        scene_tables = generate_scene_data(
            scene_idx,
            args.frames_per_scene,
            args.output,
        )
        all_scene_tables.append(scene_tables)

    merged = merge_tables(all_scene_tables)

    # Link map records to all log tokens.
    log_tokens = [log["token"] for log in merged["log"]]
    for map_record in static_tables["map"]:
        map_record["log_tokens"] = log_tokens

    version_dir = os.path.join(args.output, "v1.0-mini")
    for key, data in {**static_tables, **merged}.items():
        write_json(os.path.join(version_dir, f"{key}.json"), data)

    # Create a minimal map mask image so nuscenes-devkit can initialize.
    map_path = os.path.join(args.output, "maps", "empty.png")
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(map_path)

    print(f"[gen_dummy_nuscenes] Synthetic dataset written to {args.output}")
    print(f"  Scenes: {args.num_scenes}, frames/scene: {args.frames_per_scene}")
    print(f"  Total samples (keyframes): {len(merged['sample'])}")
    print(f"  Total camera frames: {len(merged['sample_data'])}")


if __name__ == "__main__":
    main()
