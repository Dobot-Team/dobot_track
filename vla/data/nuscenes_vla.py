import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vla.utils.common import (
    build_custom_multimodal_prompt,
    build_transform,
    preprocess_custom_images,
)


# ------------------------------------------------------------------
# Coordinate utilities
# ------------------------------------------------------------------
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])


def transform_global_to_ego(
    points_global: np.ndarray,
    current_translation: np.ndarray,
    current_rotation: np.ndarray,
) -> np.ndarray:
    """Transform points from global frame to current ego frame.

    Args:
        points_global: (N, 3) array in global coordinates.
        current_translation: (3,) current ego translation in global coords.
        current_rotation: (4,) current ego rotation quaternion (w, x, y, z).
    Returns:
        (N, 3) points in ego coordinates.
    """
    # current_rotation is ego->global; we need global->ego, i.e. R^T.
    R = quaternion_to_rotation_matrix(current_rotation)
    return (points_global - current_translation) @ R.T


def interpolate_future_positions(
    sample_data_list: List[dict],
    ego_poses: List[dict],
    current_time: int,
    n_future: int,
    dt: float,
) -> np.ndarray:
    """Linearly interpolate future ego positions at fixed temporal intervals.

    Args:
        sample_data_list: List of future sample_data records (starting from current).
        ego_poses: Corresponding ego_pose records.
        current_time: Timestamp of the current frame (microseconds).
        n_future: Number of future points.
        dt: Temporal spacing in seconds.
    Returns:
        (n_future, 3) array of global positions.
    """
    times = np.array([sd["timestamp"] for sd in sample_data_list], dtype=np.float64)
    translations = np.array([ep["translation"] for ep in ego_poses], dtype=np.float32)

    target_times = current_time + np.arange(1, n_future + 1) * dt * 1e6
    positions = np.zeros((n_future, 3), dtype=np.float32)

    for i, tt in enumerate(target_times):
        idx = np.searchsorted(times, tt)
        if idx == 0:
            positions[i] = translations[0]
        elif idx >= len(times):
            positions[i] = translations[-1]
        else:
            t0, t1 = times[idx - 1], times[idx]
            alpha = (tt - t0) / (t1 - t0) if t1 != t0 else 0.0
            positions[i] = (1.0 - alpha) * translations[idx - 1] + alpha * translations[idx]
    return positions


# ------------------------------------------------------------------
# NuScenes trajectory builder
# ------------------------------------------------------------------
def collect_sample_data_future(
    nusc: NuScenes,
    current_sd_token: str,
    min_future_us: float,
) -> Tuple[List[dict], List[dict]]:
    """Collect sample_data and ego_pose records forward in time."""
    sample_data_list = []
    ego_poses = []
    sd_token = current_sd_token
    while sd_token != "":
        sd = nusc.get("sample_data", sd_token)
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        sample_data_list.append(sd)
        ego_poses.append(ep)
        # Stop once we have enough future time.
        if sd["timestamp"] - sample_data_list[0]["timestamp"] >= min_future_us:
            break
        sd_token = sd["next"]
    return sample_data_list, ego_poses


def build_trajectory_for_sample(
    nusc: NuScenes,
    sample: dict,
    n_future: int = 24,
    dt: float = 0.25,
    cam_channel: str = "CAM_FRONT",
) -> Tuple[np.ndarray, List[str]]:
    """Build future trajectory in current ego frame and history image paths.

    Returns:
        traj: (n_future, 2) trajectory in current ego coordinates (x, y).
        history_paths: list of history image paths (oldest -> newest).
    """
    current_sd_token = sample["data"][cam_channel]
    current_sd = nusc.get("sample_data", current_sd_token)
    current_ep = nusc.get("ego_pose", current_sd["ego_pose_token"])
    current_time = current_sd["timestamp"]

    # Future trajectory.
    min_future_us = n_future * dt * 1e6 + 1e5  # add small buffer
    future_sds, future_eps = collect_sample_data_future(nusc, current_sd_token, min_future_us)
    future_positions_global = interpolate_future_positions(
        future_sds, future_eps, current_time, n_future, dt
    )
    future_positions_ego = transform_global_to_ego(
        future_positions_global,
        np.array(current_ep["translation"]),
        np.array(current_ep["rotation"]),
    )
    traj = future_positions_ego[:, :2].astype(np.float32)

    # History image paths: use sample keyframes (oldest first).
    history_samples = []
    s = sample
    history_samples.append(s)
    while len(history_samples) < 4 and s["prev"] != "":
        s = nusc.get("sample", s["prev"])
        history_samples.append(s)
    history_samples = history_samples[::-1]  # oldest -> newest

    history_paths = []
    for hs in history_samples:
        sd = nusc.get("sample_data", hs["data"][cam_channel])
        history_paths.append(os.path.join(nusc.dataroot, sd["filename"]))

    return traj, history_paths


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class NuScenesVLADataset(Dataset):
    """PyTorch Dataset for VLA trajectory prediction on NuScenes-mini."""

    def __init__(
        self,
        nusc: NuScenes,
        sample_tokens: List[str],
        tokenizer,
        transform: transforms.Compose,
        input_size: int = 448,
        num_history_frames: int = 4,
        n_future: int = 24,
        dt: float = 0.25,
        cam_channel: str = "CAM_FRONT",
        prompt: str = "Predict the future trajectory of the ego vehicle.",
        cache_dir: Optional[str] = None,
        num_image_tokens: int = 256,
        max_length: int = 2048,
        model_type: str = "internvl",
        processor=None,
        image_token: Optional[str] = None,
        qwen_min_pixels: int = 3136,
        qwen_max_pixels: int = 100_000,
    ):
        self.nusc = nusc
        self.sample_tokens = sample_tokens
        self.tokenizer = tokenizer
        self.transform = transform
        self.input_size = int(input_size)
        self.num_history_frames = num_history_frames
        self.n_future = n_future
        self.dt = dt
        self.cam_channel = cam_channel
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.num_image_tokens = num_image_tokens
        self.max_length = max_length
        self.model_type = model_type
        self.processor = processor
        self.image_token = image_token
        if self.model_type == "custom" and self.processor is None:
            raise ValueError("processor is required for model_type=custom")
        if self.model_type == "custom" and (
            not isinstance(self.image_token, str) or not self.image_token
        ):
            raise ValueError("image_token is required for model_type=custom")
        self.qwen_min_pixels = qwen_min_pixels
        self.qwen_max_pixels = qwen_max_pixels

        self.samples = self._build_samples()

    def _build_samples(self) -> List[dict]:
        cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir,
                f"nuscenes_vla_samples_h{self.num_history_frames}_f{self.n_future}_it{self.num_image_tokens}_ml{self.max_length}.pkl",
            )
            if os.path.exists(cache_file):
                print(f"[NuScenesVLADataset] Loading cached samples from {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        samples = []
        for token in self.sample_tokens:
            sample = self.nusc.get("sample", token)
            # Skip samples that do not have enough history.
            s = sample
            history_count = 1
            while s["prev"] != "" and history_count < self.num_history_frames:
                s = self.nusc.get("sample", s["prev"])
                history_count += 1
            if history_count < self.num_history_frames:
                continue

            try:
                traj, image_paths = build_trajectory_for_sample(
                    self.nusc,
                    sample,
                    n_future=self.n_future,
                    dt=self.dt,
                    cam_channel=self.cam_channel,
                )
                if len(image_paths) != self.num_history_frames:
                    continue
                samples.append({
                    "token": token,
                    "traj": traj,
                    "image_paths": image_paths,
                })
            except Exception as e:
                print(f"[NuScenesVLADataset] Skipping sample {token}: {e}")
                continue

        print(f"[NuScenesVLADataset] Built {len(samples)} samples.")
        if cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(samples, f)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in item["image_paths"]]

        if self.model_type == "custom":
            pixel_values = preprocess_custom_images(
                self.processor,
                images,
                self.input_size,
            )
            text = build_custom_multimodal_prompt(
                self.tokenizer,
                self.image_token,
                [self.num_image_tokens] * len(images),
                self.prompt,
            )
            tokenizer_out = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max(
                    self.max_length,
                    self.num_image_tokens * len(images) + 256,
                ),
            )
            return {
                "pixel_values": pixel_values,
                "input_ids": tokenizer_out["input_ids"].squeeze(0),
                "attention_mask": tokenizer_out["attention_mask"].squeeze(0),
                "trajectory": torch.from_numpy(item["traj"]),
                "token": item["token"],
            }

        if self.model_type == "qwen2vl":
            # Use Qwen2VLProcessor for chat-template + dynamic patch preprocessing.
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": self.prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Limit dynamic image patches to avoid OOM on high-res NuScenes images.
            # Qwen2VLImageProcessor uses `size.shortest_edge/longest_edge` as
            # min/max pixels for smart_resize.
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(self.max_length, 4096),
                size={
                    "shortest_edge": self.qwen_min_pixels,
                    "longest_edge": self.qwen_max_pixels,
                },
            )
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),  # (N_patches, 1176)
                # image_grid_thw already has shape (num_images, 3); keep it so that
                # single-image samples do not collapse to (3,) after squeeze.
                "image_grid_thw": inputs["image_grid_thw"],  # (num_images, 3)
                "input_ids": inputs["input_ids"].squeeze(0),  # (seq_len)
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "mm_token_type_ids": inputs.get(
                    "mm_token_type_ids",
                    torch.zeros_like(inputs["input_ids"]),
                ).squeeze(0),
                "trajectory": torch.from_numpy(item["traj"]),
                "token": item["token"],
            }

        # InternVL-style: manual image token placeholders + torchvision transform.
        pixel_values = torch.stack([self.transform(img) for img in images], dim=0)
        image_token_str = "<img>" + "<IMG_CONTEXT>" * self.num_image_tokens + "</img>"
        image_tokens = "\n".join([image_token_str] * len(images))
        text = f"{image_tokens}\nUser: {self.prompt}\nAssistant:"

        tokenizer_out = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": pixel_values,  # (H, C, H_img, W_img)
            "input_ids": tokenizer_out["input_ids"].squeeze(0),
            "attention_mask": tokenizer_out["attention_mask"].squeeze(0),
            "trajectory": torch.from_numpy(item["traj"]),  # (n_future, 2)
            "token": item["token"],
        }


import pytorch_lightning as pl


# ------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------
class NuScenesVLADataModule(pl.LightningDataModule):
    """PyTorch-Lightning DataModule for NuScenes VLA."""

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        tokenizer=None,
        input_size: int = 448,
        num_history_frames: int = 4,
        n_future: int = 24,
        dt: float = 0.25,
        cam_channel: str = "CAM_FRONT",
        batch_size: int = 1,
        num_workers: int = 4,
        val_split: float = 0.2,
        prompt: str = "Predict the future trajectory of the ego vehicle.",
        cache_dir: Optional[str] = None,
        seed: int = 42,
        num_image_tokens: int = 256,
        max_length: int = 2048,
        model_type: str = "internvl",
        processor=None,
        image_token: Optional[str] = None,
        qwen_min_pixels: int = 3136,
        qwen_max_pixels: int = 100_000,
    ):
        super().__init__()
        self.dataroot = dataroot
        self.version = version
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.num_history_frames = num_history_frames
        self.n_future = n_future
        self.dt = dt
        self.cam_channel = cam_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.seed = seed
        self.num_image_tokens = num_image_tokens
        self.max_length = max_length
        self.model_type = model_type
        self.processor = processor
        self.image_token = image_token
        self.qwen_min_pixels = qwen_min_pixels
        self.qwen_max_pixels = qwen_max_pixels

        self.transform = build_transform(input_size)

        self.nusc = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        print(f"[NuScenesVLADataModule] Loading NuScenes {self.version} from {self.dataroot}")
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

        # Use only samples with front camera data.
        sample_tokens = [s["token"] for s in self.nusc.sample if self.cam_channel in s["data"]]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(sample_tokens)
        n_val = max(1, int(len(sample_tokens) * self.val_split))
        train_tokens = sample_tokens[n_val:]
        val_tokens = sample_tokens[:n_val]

        common_kwargs = {
            "nusc": self.nusc,
            "tokenizer": self.tokenizer,
            "transform": self.transform,
            "input_size": self.input_size,
            "num_history_frames": self.num_history_frames,
            "n_future": self.n_future,
            "dt": self.dt,
            "cam_channel": self.cam_channel,
            "prompt": self.prompt,
            "cache_dir": self.cache_dir,
            "num_image_tokens": self.num_image_tokens,
            "max_length": self.max_length,
            "model_type": self.model_type,
            "processor": self.processor,
            "image_token": self.image_token,
            "qwen_min_pixels": self.qwen_min_pixels,
            "qwen_max_pixels": self.qwen_max_pixels,
        }

        self.train_dataset = NuScenesVLADataset(
            sample_tokens=train_tokens,
            **common_kwargs,
        )
        self.val_dataset = NuScenesVLADataset(
            sample_tokens=val_tokens,
            **common_kwargs,
        )

    def collate_fn(self, batch: List[dict]) -> dict:
        if self.model_type == "qwen2vl":
            # Qwen2-VL: each sample has variable-length pixel_values and input_ids.
            # Concatenate vision inputs; pad text inputs to max length in batch.
            pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)  # (sum N_patches, 1176)
            image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)  # (B*H, 3)

            input_ids_list = [b["input_ids"] for b in batch]
            attention_mask_list = [b["attention_mask"] for b in batch]
            mm_token_type_ids_list = [b["mm_token_type_ids"] for b in batch]
            max_len = max(x.shape[0] for x in input_ids_list)
            pad_token_id = self.tokenizer.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.tokenizer.eos_token_id

            input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            mm_token_type_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, (ids, mask, mm) in enumerate(zip(input_ids_list, attention_mask_list, mm_token_type_ids_list)):
                L = ids.shape[0]
                input_ids[i, :L] = ids
                attention_mask[i, :L] = mask
                mm_token_type_ids[i, :L] = mm

            trajectories = torch.stack([b["trajectory"] for b in batch])
            tokens = [b["token"] for b in batch]

            return {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_token_type_ids,
                "trajectories": trajectories,
                "tokens": tokens,
            }

        # Fixed-size images (or Qwen3-VL's fixed per-image patch matrices).
        pixel_values = torch.stack([b["pixel_values"] for b in batch]).flatten(0, 1)

        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        trajectories = torch.stack([b["trajectory"] for b in batch])
        tokens = [b["token"] for b in batch]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trajectories": trajectories,
            "tokens": tokens,
        }

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


def split_scenes(nusc: NuScenes, val_split: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split scene tokens into train/val."""
    scene_tokens = [s["token"] for s in nusc.scene]
    rng = np.random.default_rng(seed)
    rng.shuffle(scene_tokens)
    n_val = max(1, int(len(scene_tokens) * val_split))
    return scene_tokens[n_val:], scene_tokens[:n_val]
