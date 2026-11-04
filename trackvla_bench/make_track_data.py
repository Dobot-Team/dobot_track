#!/usr/bin/env python3
"""
用途: 将原始跟踪/仿真数据整理成 TrackVLA 训练格式，打包帧图、mask 和 JSONL 样本。
使用: `python tools/make_track_data.py --input_root <raw_data_root> --output_root <out_root> [--only_success] [--instruction "..."] [--horizon 8] [--dt 0.1]`。
输出: `frames/`、`masks/`、`jsonl/` 和 `dataset.json`。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import math
import zipfile  # 新增：引入 zipfile 模块
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict


@dataclass
class EpisodePaths:
    seed_dir: Path
    run_dir: Path
    stem: str
    mp4: Optional[Path]
    mask_mp4: Optional[Path]
    info_json: Path


def find_ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_sorted_images(directory: Path) -> List[Path]:
    image_paths = [p for p in directory.glob("*.jpg")]
    image_paths.sort(key=lambda p: natural_sort_key(p.name))
    return image_paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_ffmpeg(ffmpeg_path: str, mp4_path: Path, out_dir: Path, quality: int = 2) -> List[Path]:
    ensure_dir(out_dir)
    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(mp4_path),
        "-q:v", str(quality),
        str(pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list_sorted_images(out_dir)


def load_episode_info(info_json_path: Path) -> List[dict]:
    with open(info_json_path, "r") as f:
        data = json.load(f)
    return data


def load_episode_status(status_json_path: Path) -> Optional[dict]:
    try:
        with open(status_json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def build_actions_from_info(steps: List[dict]) -> List[List[float]]:
    actions: List[List[float]] = []
    for step in steps:
        base_velocity = step.get("base_velocity") or [0.0, 0.0, 0.0]
        fwd, lat, yaw = float(base_velocity[0]), float(base_velocity[1]), float(base_velocity[2])
        actions.append([fwd, lat, yaw])
    return actions


def integrate_future_trajectory(
    actions: List[List[float]], start_index: int, horizon: int, dt: float = 1.0
) -> List[List[float]]:
    x, y, theta = 0.0, 0.0, 0.0
    trajectory: List[List[float]] = []
    if horizon <= 0 or start_index >= len(actions):
        return trajectory
    end_index = min(len(actions) - 1, start_index + horizon)
    for k in range(start_index, end_index + 1):
        vx, vy, wz = actions[k]
        dx_global = vx * math.cos(theta) - vy * math.sin(theta)
        dy_global = vx * math.sin(theta) + vy * math.cos(theta)
        x += dx_global * dt
        y += dy_global * dt
        theta += wz * dt
        trajectory.append([x, y, theta])
    return trajectory


def collect_episode_pairs(input_root: Path) -> List[EpisodePaths]:
    episodes: List[EpisodePaths] = []
    for seed_dir in sorted(input_root.glob("seed_*")):
        if not seed_dir.is_dir(): continue
        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir(): continue
            for info_json in sorted(run_dir.glob("*_info.json")):
                stem = info_json.name[:-10]
                mp4 = (run_dir / f"{stem}.mp4")
                mask_mp4 = (run_dir / f"{stem}_mask.mp4")
                episodes.append(
                    EpisodePaths(
                        seed_dir=seed_dir, run_dir=run_dir, stem=stem,
                        mp4=mp4 if mp4.exists() else None,
                        mask_mp4=mask_mp4 if mask_mp4.exists() else None,
                        info_json=info_json,
                    )
                )
    return episodes


def should_keep_episode(run_dir: Path, stem: str, only_success: bool) -> bool:
    if not only_success: return True
    status = load_episode_status(run_dir / f"{stem}.json")
    if not status: return False
    success_val = status.get("success")
    finish = status.get("finish")
    following_rate = status.get("following_rate", 0.0)
    status_str = str(status.get("status", "")).lower()
    return (bool(finish) and following_rate >= 0.8) or \
           (isinstance(success_val, (int, float)) and success_val > 0 and following_rate >= 0.35) or \
           ("success" in status_str)


def main():
    parser = argparse.ArgumentParser(description="Make TrackVLA training data with Masks (ZIP based)")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    frames_root = output_root / "frames"
    masks_root = output_root / "masks"
    jsonl_root = output_root / "jsonl"
    
    [ensure_dir(d) for d in [frames_root, masks_root, jsonl_root]]
    
    ffmpeg_path = find_ffmpeg_executable()
    if not ffmpeg_path: raise RuntimeError("ffmpeg not found.")

    episodes = collect_episode_pairs(Path(args.input_root))
    all_samples = []
    valid_episode_count = 0 

    for ep in episodes:
        if not should_keep_episode(ep.run_dir, ep.stem, args.only_success): continue
        if ep.mp4 is None or ep.mask_mp4 is None: continue 

        # ==========================================
        # 1. 提取并处理原图 (打包为 ZIP，后缀依然叫 .jpgs)
        # ==========================================
        temp_frames_dir = frames_root / ep.seed_dir.name / ep.run_dir.name / f"{ep.stem}_temp"
        frame_paths = extract_frames_ffmpeg(ffmpeg_path, ep.mp4, temp_frames_dir)
        if not frame_paths: continue

        abs_frames_file_dir = frames_root / ep.seed_dir.name / ep.run_dir.name
        ensure_dir(abs_frames_file_dir)
        frames_save_path = abs_frames_file_dir / f"{ep.stem}.jpgs"
        
        # 使用 ZIP_STORED (不压缩)，因为 jpeg 已经是压缩格式了，避免训练读取时额外消耗 CPU 解压
        with zipfile.ZipFile(frames_save_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for fp in frame_paths:
                # 直接将文件写入 zip 中，arcname 是在 zip 包内的虚拟文件名
                zf.write(fp, arcname=fp.name)
        
        # 清理临时原图目录
        shutil.rmtree(temp_frames_dir)

        # ==========================================
        # 2. 提取并处理 Mask (打包为 ZIP，后缀依然叫 .masks)
        # ==========================================
        temp_mask_dir = masks_root / ep.seed_dir.name / ep.run_dir.name / f"{ep.stem}_mask_temp"
        mask_frame_paths = extract_frames_ffmpeg(ffmpeg_path, ep.mask_mp4, temp_mask_dir)
        
        pixel_counts: Dict[str, int] = {}
        abs_mask_file_dir = masks_root / ep.seed_dir.name / ep.run_dir.name
        ensure_dir(abs_mask_file_dir)
        mask_save_path = abs_mask_file_dir / f"{ep.stem}.masks" 
        
        with zipfile.ZipFile(mask_save_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for mp in mask_frame_paths:
                # 将文件写入 zip 中
                zf.write(mp, arcname=mp.name)
                
                # 计算 pixel_count (此处依然需要加载进内存计算一下)
                with Image.open(mp) as img:
                    mask_tensor = torch.from_numpy(np.array(img.convert("L"))).float() / 255.0
                    pixel_counts[mp.name] = int(torch.sum(mask_tensor > 0.5).item())
        
        # 清理临时 mask 目录
        shutil.rmtree(temp_mask_dir)

        # ==========================================
        # 3. 构建样本
        # ==========================================
        steps = load_episode_info(ep.info_json)
        actions_full = build_actions_from_info(steps)
        status = load_episode_status(ep.run_dir / f"{ep.stem}.json")
        
        desired_len = len(frame_paths)
        if len(actions_full) < desired_len:
            actions_full = actions_full + [actions_full[-1]] * (desired_len - len(actions_full))

        instruction_text = status.get("instruction", "Track the target person without collision.")
        
        episode_samples = []
        for j in range(len(frame_paths)):
            horizon = args.horizon
            if j + horizon > len(actions_full) - 1: continue

            current_frame_name = frame_paths[j].name
            step_info = steps[j] if j < len(steps) else {}
            
            sample = {
                "frames_file": str(frames_save_path.relative_to(output_root).as_posix()), 
                "current": current_frame_name, 
                "mask_file": str(mask_save_path.relative_to(output_root).as_posix()), 
                "pixel_count": pixel_counts.get(current_frame_name, 0), 
                "instruction": instruction_text,
                "trajectory": integrate_future_trajectory(actions_full, j, horizon, args.dt),
                "actions": actions_full[j : j + horizon + 1],
                "dt": float(args.dt),
                "collision": bool(step_info.get("collision", False)),
                "facing": float(step_info.get("facing", 1.0)),
                "target_polar": {
                    "angle_rad": float(step_info.get("angle_rad", 0.0)),
                    "angle_deg": float(step_info.get("angle_deg", 0.0)),
                    "distance_2d": float(step_info.get("distance_2d", 0.0))
                },
                "target_distance": float(step_info.get("target_distance", step_info.get("dis_to_human", 0.0)))
            }
            episode_samples.append(sample)

        if episode_samples:
            valid_episode_count += 1
            
            abs_jsonl_dir = jsonl_root / ep.seed_dir.name / ep.run_dir.name
            ensure_dir(abs_jsonl_dir)
            with open(abs_jsonl_dir / f"{ep.stem}.jsonl", "w") as f:
                for s in episode_samples:
                    f.write(json.dumps(s) + "\n")
            all_samples.extend(episode_samples)

    with open(output_root / "dataset.json", "w") as f:
        json.dump(all_samples, f)
    
    print("Done.")
    print(f"Total valid episodes processed: {valid_episode_count}")
    print(f"Total samples (objects) saved: {len(all_samples)}")
    print(f"Packed frames saved to: {frames_root}")
    print(f"Masks saved to: {masks_root}")


if __name__ == "__main__":
    main()

    # sim_data/train_split/dt_train_3 写shell
    # python tools/make_tracking_data.py --input_root sim_data/stt_train_5 --output_root data/sample/stt_train_70 --only_success
