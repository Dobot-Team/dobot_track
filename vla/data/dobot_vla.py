import hashlib
import json
import multiprocessing as mp
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vla.utils.common import (
    build_custom_multimodal_prompt,
    build_transform,
    preprocess_custom_images,
)

# Allow loading slightly truncated JPEGs without crashing the training loop.
# The dataset still logs the path so that the original can be re-copied if needed.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------------------------------------------------
# Coordinate / trajectory utilities
# ------------------------------------------------------------------
def interpolate_pose(t: float, labels: List[dict]) -> np.ndarray:
    """Linearly interpolate (x, y, yaw) from labels at timestamp t (seconds).

    Args:
        t: Target timestamp in seconds.
        labels: List of label dicts with "timestamp" and "pose" keys.
    Returns:
        (3,) array of [x, y, yaw].
    """
    ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)
    idx = int(np.searchsorted(ts, t))
    if idx == 0:
        p = labels[0]["pose"]
        return np.array([p["x"], p["y"], p["yaw"]], dtype=np.float32)
    if idx >= len(labels):
        p = labels[-1]["pose"]
        return np.array([p["x"], p["y"], p["yaw"]], dtype=np.float32)

    t0, t1 = ts[idx - 1], ts[idx]
    alpha = float((t - t0) / (t1 - t0)) if t1 != t0 else 0.0

    p0 = labels[idx - 1]["pose"]
    p1 = labels[idx]["pose"]
    x = (1.0 - alpha) * p0["x"] + alpha * p1["x"]
    y = (1.0 - alpha) * p0["y"] + alpha * p1["y"]

    # Yaw short-path interpolation.
    y0, y1 = p0["yaw"], p1["yaw"]
    dy = (y1 - y0 + np.pi) % (2.0 * np.pi) - np.pi
    yaw = y0 + alpha * dy
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([x, y, yaw], dtype=np.float32)


def transform_to_ego(
    current_pose: np.ndarray,
    future_poses: np.ndarray,
) -> np.ndarray:
    """Transform future poses into the current ego frame.

    The ego frame convention used here:
      - x_ego points forward (along the robot's heading)
      - y_ego points to the robot's left
      - yaw_ego is the relative heading w.r.t. current heading

    Args:
        current_pose: (3,) [x, y, yaw] in bag coordinate frame.
        future_poses: (N, 3) [x, y, yaw] in bag coordinate frame.
    Returns:
        (N, 3) [x_ego, y_ego, yaw_ego].
    """
    dx = future_poses[:, 0] - current_pose[0]
    dy = future_poses[:, 1] - current_pose[1]
    cos_yaw = np.cos(current_pose[2])
    sin_yaw = np.sin(current_pose[2])

    ego_x = dx * cos_yaw + dy * sin_yaw
    ego_y = -dx * sin_yaw + dy * cos_yaw
    ego_yaw = future_poses[:, 2] - current_pose[2]
    ego_yaw = (ego_yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack([ego_x, ego_y, ego_yaw], axis=-1).astype(np.float32)


def find_nearest_frame_idx(target_ts: float, frame_ts: np.ndarray) -> int:
    """Return index of the frame whose timestamp is closest to target_ts."""
    return int(np.argmin(np.abs(frame_ts - target_ts)))


def _augment_scene_desc(scene_desc: str, rng: Optional[np.random.Generator] = None) -> str:
    """Paraphrase the person-following scene_desc with varied syntax.

    The original descriptions follow a fixed pattern like:
      "跟着前面的人走，目标人物特征为男性，黑色短发，..."
    This function randomly picks a template so the model sees the same
    semantic instruction expressed in different word orders.
    """
    if not scene_desc:
        return ""

    templates = [
        "{scene_desc}",
        "任务：跟随前方人物。{scene_desc}",
        "请跟随目标人物。{scene_desc}",
        "目标人物在前方，{scene_desc}",
        "跟随任务：{scene_desc}",
        "前方有需要跟随的人物，{scene_desc}",
        "当前任务为跟随前方行人。{scene_desc}",
        "注意前方目标人物，{scene_desc}",
    ]
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(templates).format(scene_desc=scene_desc)


def _validate_video_frames(frames: List[dict], cam_channel: str) -> set:
    """Return indices of frames whose cam_channel image can be opened cleanly.

    LOAD_TRUNCATED_IMAGES is temporarily disabled so that truncated JPEGs are
    rejected at sample-build time; the global flag remains enabled for training.
    """
    old_load_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    valid_idxs = set()
    try:
        for idx, frame in enumerate(frames):
            p = frame.get(cam_channel)
            if p is None or not os.path.exists(p):
                continue
            try:
                with Image.open(p) as img:
                    img.load()
                valid_idxs.add(idx)
            except Exception:
                # Skip corrupt/missing frames silently during cache build.
                pass
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_load_truncated
    return valid_idxs


def _build_samples_for_video_worker(args: Tuple) -> List[dict]:
    """Picklable worker used for parallel sample-cache construction."""
    (
        json_path,
        num_history_frames,
        history_duration,
        n_future,
        future_dt,
        cam_channel,
        sample_interval,
    ) = args

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"[DobotVLADataset] Skipping {json_path}: {e}")
        return []

    frames = data.get("frames", [])
    labels = data.get("labels", [])
    if len(frames) == 0 or len(labels) == 0:
        return []

    valid_frame_idxs = _validate_video_frames(frames, cam_channel)

    frame_ts = np.array([f["timestamp"] * 1e-9 for f in frames], dtype=np.float64)
    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)

    history_offsets = np.linspace(-history_duration, 0.0, num_history_frames)
    future_offsets = np.arange(1, n_future + 1) * future_dt
    last_label_t = label_ts[-1]

    video_samples = []
    for i in range(0, len(labels), sample_interval):
        curr_t = label_ts[i]

        earliest_history_t = curr_t + history_offsets[0]
        if earliest_history_t < frame_ts[0]:
            continue

        latest_future_t = curr_t + future_offsets[-1]
        if latest_future_t > last_label_t:
            continue

        history_ts = curr_t + history_offsets
        history_frame_idxs = [find_nearest_frame_idx(t, frame_ts) for t in history_ts]

        # All history frames must be valid (existing and non-corrupt).
        if not set(history_frame_idxs).issubset(valid_frame_idxs):
            continue

        history_paths = [frames[idx][cam_channel] for idx in history_frame_idxs]
        video_samples.append({
            "json_path": json_path,
            "label_idx": i,
            "history_paths": history_paths,
        })

    return video_samples


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class DobotVLADataset(Dataset):
    """VLA dataset for Dobot post-sync bags.

    Each training.json is treated as one video. Samples are drawn at equal
    intervals from the labels. For each sample:
      - 4 history frames are selected by timestamp (nearest match).
      - 24 future (x, y, yaw) points are interpolated at 0.25s spacing and
        transformed into the current ego frame.
    """

    def __init__(
        self,
        json_paths: List[str],
        tokenizer,
        transform: transforms.Compose,
        input_size: int = 448,
        num_history_frames: int = 4,
        history_duration: float = 1.0,
        n_future: int = 24,
        future_dt: float = 0.25,
        cam_channel: str = "cam1",
        sample_interval: int = 5,
        prompt: str = "Predict the future trajectory of the ego vehicle.",
        num_image_tokens: int = 256,
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
        num_build_workers: int = 1,
        use_scene_desc: bool = True,
        scene_desc_augment: bool = True,
        scene_desc_seed: int = 42,
        model_type: str = "internvl",
        processor=None,
        image_token: Optional[str] = None,
    ):
        self.json_paths = json_paths
        self.tokenizer = tokenizer
        self.transform = transform
        self.input_size = int(input_size)
        self.num_history_frames = num_history_frames
        self.history_duration = history_duration
        self.n_future = n_future
        self.future_dt = future_dt
        self.cam_channel = cam_channel
        self.sample_interval = sample_interval
        self.prompt = prompt
        self.num_image_tokens = num_image_tokens
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.num_build_workers = num_build_workers
        self.use_scene_desc = use_scene_desc
        self.scene_desc_augment = scene_desc_augment
        self.scene_desc_seed = scene_desc_seed
        self.model_type = model_type
        self.processor = processor
        self.image_token = image_token
        if self.model_type == "custom" and self.processor is None:
            raise ValueError("processor is required for model_type=custom")
        if self.model_type == "custom" and (
            not isinstance(self.image_token, str) or not self.image_token
        ):
            raise ValueError("image_token is required for model_type=custom")

        self._video_cache: Dict[str, dict] = {}
        self._scene_desc_cache: Dict[str, str] = {}
        self.samples = self._build_samples()

    def _get_video_data(self, json_path: str) -> dict:
        """Load (and cache) the full training.json for a video.

        Caching avoids re-reading the same json from storage for every sample,
        which is a major bottleneck when the data lives on a slow mount.
        """
        if json_path not in self._video_cache:
            with open(json_path) as f:
                self._video_cache[json_path] = json.load(f)
        return self._video_cache[json_path]

    def _get_scene_desc(self, json_path: str) -> str:
        """Return (and cache) the scene_desc for a video from bag_meta."""
        if json_path not in self._scene_desc_cache:
            data = self._get_video_data(json_path)
            bag_meta = data.get("bag_meta", {})
            self._scene_desc_cache[json_path] = bag_meta.get("scene_desc", "")
        return self._scene_desc_cache[json_path]

    def _build_samples(self) -> List[dict]:
        cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Hash the sorted json path list so train/val splits get separate
            # caches and changing the video list invalidates the old cache.
            path_hash = hashlib.md5(
                "\n".join(sorted(self.json_paths)).encode("utf-8")
            ).hexdigest()[:8]
            cache_file = os.path.join(
                self.cache_dir,
                f"dobot_vla_samples_h{self.num_history_frames}_f{self.n_future}_"
                f"si{self.sample_interval}_cam{self.cam_channel}_ph{path_hash}.pkl",
            )
            if os.path.exists(cache_file):
                print(f"[DobotVLADataset] Loading cached samples from {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        worker_args = [
            (
                json_path,
                self.num_history_frames,
                self.history_duration,
                self.n_future,
                self.future_dt,
                self.cam_channel,
                self.sample_interval,
            )
            for json_path in self.json_paths
        ]

        if self.num_build_workers > 1:
            with mp.Pool(processes=self.num_build_workers) as pool:
                per_video_samples = pool.map(_build_samples_for_video_worker, worker_args)
        else:
            per_video_samples = [_build_samples_for_video_worker(args) for args in worker_args]

        samples = []
        for video_samples in per_video_samples:
            samples.extend(video_samples)

        print(f"[DobotVLADataset] Built {len(samples)} samples from {len(self.json_paths)} videos.")
        if cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(samples, f)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        json_path = item["json_path"]
        label_idx = item["label_idx"]

        data = self._get_video_data(json_path)
        frames = data["frames"]
        labels = data["labels"]

        curr_label = labels[label_idx]
        curr_t = curr_label["timestamp"]
        curr_pose = np.array([
            curr_label["pose"]["x"],
            curr_label["pose"]["y"],
            curr_label["pose"]["yaw"],
        ], dtype=np.float32)

        # History images.
        images = [Image.open(p).convert("RGB") for p in item["history_paths"]]
        if self.model_type == "custom":
            pixel_values = preprocess_custom_images(
                self.processor,
                images,
                self.input_size,
            )
        else:
            pixel_values = torch.stack([self.transform(img) for img in images], dim=0)

        # Future trajectory in ego frame.
        future_ts = curr_t + np.arange(1, self.n_future + 1) * self.future_dt
        future_poses = np.array([interpolate_pose(t, labels) for t in future_ts], dtype=np.float32)
        traj_ego = transform_to_ego(curr_pose, future_poses)

        # Build text prompt. If scene_desc is available, prepend it (with
        # optional syntactic augmentation) so the model knows whether this is a
        # person-following task or a regular navigation task.
        effective_prompt = self.prompt
        if self.use_scene_desc:
            scene_desc = self._get_scene_desc(json_path)
            if scene_desc:
                if self.scene_desc_augment:
                    # Deterministic per-sample RNG so augmentation works safely
                    # with multiprocessing DataLoader workers.
                    token_str = f"{json_path}#{label_idx}#{self.scene_desc_seed}"
                    rng = np.random.default_rng(abs(hash(token_str)) % (2**32))
                    scene_desc = _augment_scene_desc(scene_desc, rng)
                effective_prompt = f"{scene_desc} {self.prompt}"

        if self.model_type == "custom":
            text = build_custom_multimodal_prompt(
                self.tokenizer,
                self.image_token,
                [self.num_image_tokens] * len(images),
                effective_prompt,
            )
        else:
            # InternVL-style text prompt.
            image_token_str = "<img>" + "<IMG_CONTEXT>" * self.num_image_tokens + "</img>"
            image_tokens = "\n".join([image_token_str] * len(images))
            text = f"{image_tokens}\nUser: {effective_prompt}\nAssistant:"

        tokenizer_max_length = self.max_length
        if self.model_type == "custom":
            tokenizer_max_length = max(
                tokenizer_max_length,
                self.num_image_tokens * len(images) + 256,
            )
        tokenizer_out = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer_max_length,
        )

        return {
            "pixel_values": pixel_values,  # (H, C, H_img, W_img)
            "input_ids": tokenizer_out["input_ids"].squeeze(0),
            "attention_mask": tokenizer_out["attention_mask"].squeeze(0),
            "trajectory": torch.from_numpy(traj_ego),  # (n_future, 3)
            "token": f"{os.path.basename(json_path)}#{label_idx}",
        }


# ------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------
class DobotVLADataModule(pl.LightningDataModule):
    """PyTorch-Lightning DataModule for Dobot VLA training data."""

    def __init__(
        self,
        training_json_list: str,
        tokenizer,
        processor=None,
        model_type: str = "internvl",
        image_token: Optional[str] = None,
        input_size: int = 448,
        num_history_frames: int = 4,
        history_duration: float = 1.0,
        n_future: int = 24,
        future_dt: float = 0.25,
        cam_channel: str = "cam1",
        batch_size: int = 1,
        num_workers: int = 4,
        val_split: float = 0.1,
        sample_interval: int = 5,
        prompt: str = "Predict the future trajectory of the ego vehicle.",
        num_image_tokens: int = 256,
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        num_build_workers: int = 1,
        use_scene_desc: bool = True,
        scene_desc_augment: bool = True,
    ):
        super().__init__()
        self.training_json_list = training_json_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type
        self.image_token = image_token
        self.input_size = input_size
        self.num_history_frames = num_history_frames
        self.history_duration = history_duration
        self.n_future = n_future
        self.future_dt = future_dt
        self.cam_channel = cam_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.sample_interval = sample_interval
        self.prompt = prompt
        self.num_image_tokens = num_image_tokens
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.seed = seed
        self.num_build_workers = num_build_workers
        self.use_scene_desc = use_scene_desc
        self.scene_desc_augment = scene_desc_augment

        self.transform = build_transform(input_size)

        self.all_json_paths = []
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DobotVLADataModule] Loading json list from {self.training_json_list}")
        with open(self.training_json_list) as f:
            self.all_json_paths = [line.strip() for line in f if line.strip()]

        # Split by video (json file) to avoid leakage.
        rng = np.random.default_rng(self.seed)
        shuffled = self.all_json_paths.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * self.val_split))
        train_jsons = shuffled[n_val:]
        val_jsons = shuffled[:n_val]

        print(f"[DobotVLADataModule] Train videos: {len(train_jsons)}, Val videos: {len(val_jsons)}")

        common_kwargs = {
            "tokenizer": self.tokenizer,
            "transform": self.transform,
            "input_size": self.input_size,
            "num_history_frames": self.num_history_frames,
            "history_duration": self.history_duration,
            "n_future": self.n_future,
            "future_dt": self.future_dt,
            "cam_channel": self.cam_channel,
            "sample_interval": self.sample_interval,
            "prompt": self.prompt,
            "num_image_tokens": self.num_image_tokens,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "num_build_workers": self.num_build_workers,
            "use_scene_desc": self.use_scene_desc,
            "scene_desc_augment": self.scene_desc_augment,
            "scene_desc_seed": self.seed,
            "model_type": self.model_type,
            "processor": self.processor,
            "image_token": self.image_token,
        }

        self.train_dataset = DobotVLADataset(json_paths=train_jsons, **common_kwargs)
        self.val_dataset = DobotVLADataset(json_paths=val_jsons, **common_kwargs)

    def collate_fn(self, batch: List[dict]) -> dict:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        pixel_values = pixel_values.flatten(0, 1)

        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        trajectories = torch.stack([b["trajectory"] for b in batch])  # (B, n_future, 3)
        tokens = [b["token"] for b in batch]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trajectories": trajectories,
            "tokens": tokens,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
