#!/usr/bin/env python3
"""Evaluate an RGB-only TrackVLA checkpoint in TrackOpen's Habitat benchmark.

This runner deliberately keeps the simulator dependency outside of embodvla.
``--trackvla-root`` accepts either a full TrackOpen checkout or an asset pack
containing ``data/``.  When the latter is used, ``--trackvla-code-root``
selects the checkout that supplies Habitat, Habitat-Sim, and the custom
``evt_bench`` registrations.  Model structure is reconstructed exclusively
from the nearest checkpoint ``config.yaml``; checkpoint contents supply only
the trainable VLA state.  This lets the runner use DeepSpeed states converted
with ``zero_to_fp32.py`` without modifying their contents.

Example:
    conda run -n evt_bench python scripts/eval_trackvla_habitat.py \
        --checkpoint outputs/trackvla/checkpoints/last.ckpt \
        --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml \
        --split-num 1 --split-id 0 --save-path outputs/trackvla_habitat_eval

Without ``--config``, the runner finds the nearest ``config.yaml`` in the
checkpoint directory or one of its parents.

Pass ``--show-masks`` to overlay available SAM2 mask predictions in saved
episode videos. The option has no effect when the checkpoint has no mask task.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla.utils.history_boxes import HistoryBoxCache, resize_pad_boxes
from vla.utils.checkpoint_config import (
    checkpoint_config_path,
    load_checkpoint_config,
)
from vla.utils.common import (
    CudaStandardViTTransform,
    ResizeLongestSideAndPad,
    apply_qwen_vl_chat_template,
    build_custom_multimodal_prompt,
    build_transform,
    check_cuda_vit_transform_compatibility,
    preprocess_custom_images,
    resize_longest_side_and_pad_geometry,
)
from vla.config import (
    ExecutionStage,
    ModelExtensionContext,
    ModelExtensionsConfig,
)
from vla.utils.trainable_state_dict import (
    require_complete_trainable_state_dict,
)


DEFAULT_ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
)

# Class order is defined by vla/data/trackvla.py::build_auxiliary_targets.
# Keep these names in the evaluator so JSON outputs are human-readable rather
# than exposing only argmax integer IDs.
VQA_CLASS_NAMES = {
    "position": ("left", "center", "right"),
    "scale": ("small", "medium", "large"),
    "distance": ("near", "medium", "far"),
    "facing": ("not-facing", "facing"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an RGB-only TrackVLA checkpoint in TrackOpen Habitat."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional training YAML override; defaults to the nearest config.yaml "
            "in or above the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning checkpoint or converted DeepSpeed state-dict file.",
    )
    parser.add_argument(
        "--trackvla-root",
        default=os.environ.get("TRACKVLA_ROOT", str(PROJECT_ROOT / "trackvla_bench")),
        help=(
            "TrackOpen checkout or asset-pack root containing data/. A full checkout "
            "continues to work as before."
        ),
    )
    parser.add_argument(
        "--trackvla-code-root",
        default=os.environ.get("TRACKVLA_CODE_ROOT"),
        help=(
            "TrackOpen checkout containing habitat-lab and evt_bench when "
            "--trackvla-root is an asset pack."
        ),
    )
    parser.add_argument(
        "--trackvla-assets-root",
        default=os.environ.get("TRACKVLA_ASSETS_ROOT"),
        help=(
            "Optional asset-pack root containing data/. Overrides assets discovered "
            "from --trackvla-root."
        ),
    )
    parser.add_argument(
        "--exp-config",
        default="habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml",
        help="Habitat config path, relative to the TrackOpen code root when not absolute.",
    )
    parser.add_argument("--split-id", required=True, type=int)
    parser.add_argument("--split-num", required=True, type=int)
    parser.add_argument(
        "--save-path",
        required=True,
        help="Directory for per-episode JSON and optional MP4 files.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model.model_path from the YAML, for example a local HF directory.",
    )
    parser.add_argument(
        "--vision-model-path",
        default=None,
        help="Override model.vision_model_path for model_type=custom.",
    )
    parser.add_argument(
        "--llm-model-path",
        default=None,
        help="Override model.llm_model_path for model_type=custom.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when visible, otherwise cpu.",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow the base VLM to download when it is absent from the local HF cache.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.1,
        help="TrackVLA trajectory integration interval used to recover the action.",
    )
    parser.add_argument(
        "--waypoint-index",
        type=int,
        default=0,
        help="Predicted waypoint used as the command. TrackVLA labels use index 0.",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="Run VLA every N simulator steps and repeat its last action between calls.",
    )
    parser.add_argument(
        "--no-command-clip",
        action="store_true",
        help="Do not clip predicted normalized commands to [-1, 1] before Habitat.",
    )
    parser.add_argument(
        "--rgb-sensor",
        default="agent_1_articulated_agent_jaw_rgb",
        help="RGB observation key for the follower robot.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save one MP4 per episode from --rgb-sensor.",
    )
    parser.add_argument(
        "--show-masks",
        action="store_true",
        help="Overlay available SAM2 masks in saved video frames.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Playback frame rate for saved MP4 files, default: 10.",
    )
    parser.add_argument(
        "--robot-action",
        default="agent_1_base_velocity",
        help="TrackOpen action name used for the follower robot.",
    )
    parser.add_argument(
        "--robot-action-arg",
        default="agent_1_base_vel",
        help="TrackOpen action argument name for the normalized command.",
    )
    parser.add_argument(
        "--action-names",
        default=",".join(DEFAULT_ACTION_NAMES),
        help="Comma-separated action names passed to TrackEnv each simulator step.",
    )
    parser.add_argument(
        "--default-instruction",
        default="Track the target person without collision.",
        help="Instruction used only when an episode does not provide one.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=4.0,
        help="Stop an episode after this robot-to-target distance persists.",
    )
    parser.add_argument(
        "--max-lost-steps",
        type=int,
        default=20,
        help="Consecutive over-distance steps before marking the episode Lost.",
    )
    parser.add_argument(
        "--continue-on-collision",
        action="store_true",
        help="Continue after TrackOpen's human_collision metric instead of stopping.",
    )
    parser.add_argument(
        "--allow-uneven-splits",
        action="store_true",
        help="Keep remainder episodes when splitting the dataset instead of legacy behavior.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print one step line every N simulator steps; 0 disables step logging.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional Habitat overrides, e.g. habitat.simulator.seed=0.",
    )
    return parser.parse_args()


def resolve_existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def dtype_from_config(value: Any) -> torch.dtype:
    normalized = str(value or "bf16").lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported model dtype: {value}")
    return mapping[normalized]


def resolve_model_source(model_path: str, allow_hf_download: bool) -> str:
    """Turn an HF repository ID into a local snapshot path before model loading.

    TrackOpen's environment currently has a newer Transformers release which
    may issue a metadata request for a repository ID even with
    ``local_files_only=True``.  Passing the cached snapshot directory avoids
    that request and keeps offline evaluation genuinely offline.
    """

    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model_path,
            local_files_only=not allow_hf_download,
        )
    except Exception as exc:
        mode = "download" if allow_hf_download else "find in the local Hugging Face cache"
        raise RuntimeError(
            f"Could not {mode} base model {model_path!r}. Supply --model-path with "
            "a local model directory, or use --allow-hf-download."
        ) from exc
    return str(Path(snapshot).resolve())


def configure_hf_modules_cache() -> None:
    """Use a writable cache for custom model code loaded with trust_remote_code."""

    cache_dir = Path(
        os.environ.get("HF_MODULES_CACHE", str(PROJECT_ROOT / "cache" / "hf_modules"))
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache_dir.resolve())


def load_vla_model(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    model_path_override: Optional[str],
    vision_model_path_override: Optional[str],
    llm_model_path_override: Optional[str],
    device: torch.device,
    allow_hf_download: bool,
) -> Tuple[torch.nn.Module, str, torch.dtype]:
    model_config = config["model"]
    model_type = str(model_config.get("model_type", "internvl")).lower()
    model_extensions = ModelExtensionsConfig.from_model_mapping(model_config)
    model_extensions.validate_for(
        context=ModelExtensionContext(
            stage=ExecutionStage.ONLINE_EVAL,
            dataset_type="trackvla",
            model_type=model_type,
            dense_tokens_per_frame=config["data"].get(
                "num_image_tokens",
                256,
            ),
        ),
    )
    sam2_config = model_extensions.sam2
    history_compressor_config = model_extensions.history_compressor

    configure_hf_modules_cache()
    num_queries = int(model_config.get("num_queries", 0))
    n_future = int(config["data"]["n_future"])
    action_head_dims = tuple(model_config.get("action_head_dims", ()))
    action_head_type = str(model_config.get("action_head_type", "legacy"))
    if num_queries <= 0 or n_future <= 0 or not action_head_dims:
        raise ValueError(
            "model.num_queries, data.n_future, and model.action_head_dims must be configured"
        )
    dtype = dtype_from_config(model_config.get("dtype", "bf16"))
    if device.type == "cpu" and dtype != torch.float32:
        print("[sim-eval] CPU selected; loading the VLM as float32.")
        dtype = torch.float32

    common_kwargs = {
        "num_queries": num_queries,
        "n_future": n_future,
        "freeze_vision": bool(model_config.get("freeze_vision", True)),
        "freeze_llm": bool(model_config.get("freeze_llm", True)),
        "action_head_dims": action_head_dims,
        "action_head_type": action_head_type,
        "dtype": dtype,
        "local_files_only": not allow_hf_download,
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
        "sam2_config": sam2_config,
        # Preserve the auxiliary-head switches from the training checkpoint.
        # Without this argument InternVL3VLA falls back to its default config
        # (grounding/VQA disabled), even when checkpoint config.yaml enables
        # them; the evaluator then writes null bbox/VQA predictions.
        "auxiliary_tasks_config": model_config.get("auxiliary_tasks_config"),
    }
    if model_type == "custom":
        vision_model_path = (
            vision_model_path_override or model_config.get("vision_model_path")
        )
        llm_model_path = llm_model_path_override or model_config.get("llm_model_path")
        if not vision_model_path or not llm_model_path:
            raise ValueError(
                "model_type=custom requires model.vision_model_path and "
                "model.llm_model_path (or their CLI overrides)"
            )
        vision_model_path = resolve_model_source(
            str(vision_model_path), allow_hf_download
        )
        llm_model_path = resolve_model_source(
            str(llm_model_path), allow_hf_download
        )
        from vla.models.custom_vla import CustomVLA

        model = CustomVLA(
            vision_model_path=vision_model_path,
            llm_model_path=llm_model_path,
            input_size=int(config["data"].get("input_size", 448)),
            **common_kwargs,
            freeze_action_head=bool(model_config.get("freeze_action_head", False)),
            gradient_checkpointing=False,
            custom_config=model_config.get("custom"),
            history_compressor_config=history_compressor_config,
        )
        configured_tokens = config["data"].get("num_image_tokens")
        if configured_tokens is not None and int(configured_tokens) != model.image_seq_length:
            raise ValueError(
                "data.num_image_tokens does not match the custom ViT/input_size: "
                f"configured {configured_tokens}, inferred {model.image_seq_length}"
            )
        if configured_tokens is None:
            # Keep the online agent's prompt layout in sync with the resolved ViT.
            config["data"]["num_image_tokens"] = model.image_seq_length
    else:
        model_path = model_path_override or model_config.get("model_path")
        if not model_path:
            raise ValueError("No model path in --model-path or config.model_path")
        model_path = resolve_model_source(str(model_path), allow_hf_download)
        common_kwargs["model_path"] = str(model_path)

    if model_type == "internvl":
        from vla.models.internvl_vla import InternVLVLA

        model = InternVLVLA(
            **common_kwargs,
            freeze_mlp1=bool(model_config.get("freeze_mlp1", True)),
        )
    elif model_type == "internvl3":
        from vla.models.internvl3_vla import InternVL3VLA

        model = InternVL3VLA(
            **common_kwargs,
            freeze_mlp1=bool(model_config.get("freeze_mlp1", True)),
            history_compressor_config=history_compressor_config,
        )
    elif model_type == "qwen2vl":
        from vla.models.qwen2vl_vla import Qwen2VLVLA

        model = Qwen2VLVLA(
            **common_kwargs,
            freeze_action_head=bool(model_config.get("freeze_action_head", False)),
        )
    elif model_type == "qwen3vl":
        from vla.models.qwen3vl_vla import Qwen3VLVLA

        model = Qwen3VLVLA(
            **common_kwargs,
            freeze_action_head=bool(model_config.get("freeze_action_head", False)),
            history_compressor_config=history_compressor_config,
        )
    elif model_type != "custom":
        raise ValueError(f"Unsupported model_type for Habitat evaluation: {model_type}")

    # The checkpoint is intentionally opened only after the model architecture
    # has been reconstructed from config.yaml.  Do not recover or inject
    # hyperparameters here: raw zero_to_fp32 outputs contain only tensors.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint_path}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint has no usable state_dict: {checkpoint_path}")
    model_keys = set(model.state_dict())
    trainable_state: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
        if key in model_keys:
            trainable_state[key] = value
    require_complete_trainable_state_dict(model, trainable_state)
    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    print(
        "[sim-eval] Loaded "
        f"{len(trainable_state)} VLA tensors from {checkpoint_path.name}; "
        f"base tensors retained={len(missing)}, unexpected={len(unexpected)}"
    )
    model.eval().to(device)
    return model, model_type, dtype


class TrackVLAHabitatAgent:
    """Convert TrackOpen RGB observations into TrackVLA normalized commands."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        data_config: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        control_dt: float,
        waypoint_index: int,
        clip_commands: bool,
        default_instruction: str,
    ):
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if waypoint_index < 0:
            raise ValueError("waypoint_index must be non-negative")
        self.model = model
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.control_dt = float(control_dt)
        self.waypoint_index = int(waypoint_index)
        self.clip_commands = clip_commands
        self.default_instruction = default_instruction.strip()
        self.history_size = int(data_config.get("num_history_frames", 4))
        self.history_frame_interval = int(
            data_config.get("history_frame_interval", 0)
        )
        self.history_stride = self.history_frame_interval + 1
        self.input_size = int(data_config.get("input_size", 448))
        self.num_image_tokens = int(data_config.get("num_image_tokens", 256))
        self.max_length = int(data_config.get("max_length", 1200))
        self.prompt = str(
            data_config.get(
                "prompt", "Predict the follower's future trajectory in the ego coordinate frame."
            )
        ).strip()
        self.qwen_min_pixels = int(data_config.get("qwen_min_pixels", 3136))
        self.qwen_max_pixels = int(data_config.get("qwen_max_pixels", 100_000))
        if self.history_size <= 0:
            raise ValueError("data.num_history_frames must be positive")
        if self.history_frame_interval < 0:
            raise ValueError("data.history_frame_interval must be non-negative")

        model_extensions = getattr(
            self.model,
            "model_extensions",
            None,
        )
        if model_extensions is None:
            model_extensions = ModelExtensionsConfig.from_values(
                sam2=getattr(self.model, "sam2_config", None),
                history_compressor=getattr(
                    self.model,
                    "history_compressor_config",
                    None,
                ),
            )
        if not isinstance(model_extensions, ModelExtensionsConfig):
            raise TypeError(
                "model.model_extensions must be a ModelExtensionsConfig"
            )
        model_extensions.history_compressor.validate_grounding(
            getattr(getattr(self.model, "auxiliary_tasks_config", None), "grounding_enabled", False))
        self.model_extensions = model_extensions
        self.extension_requirements = self.model_extensions.validate_for(
            ModelExtensionContext(
                stage=ExecutionStage.ONLINE_EVAL,
                dataset_type="trackvla",
                model_type=self.model_type,
                dense_tokens_per_frame=self.num_image_tokens,
            )
        )
        # A sampled H-frame clip spans (H-1)*stride previous simulator steps
        # plus the current step. Retaining exactly that window makes online
        # sampling match TrackVLADataset._frame_names.
        self.history_capacity = (
            (self.history_size - 1) * self.history_stride + 1
        )
        # Keep the PIL implementation for explicit CPU inference. CUDA follows
        # the same geometry below without creating an image object per frame.
        self.cpu_transform = build_transform(self.input_size)
        self.custom_cuda_transform: Optional[CudaStandardViTTransform] = None
        if self.model_type == "custom" and self.device.type == "cuda":
            image_processor = getattr(self.model, "image_processor", None)
            vision_adapter = str(getattr(self.model, "vision_adapter", "standard"))
            compatibility = check_cuda_vit_transform_compatibility(
                image_processor,
                vision_adapter=vision_adapter,
            )
            if not compatibility.supported:
                raise ValueError(
                    "The custom ViT cannot use required CUDA preprocessing: "
                    + compatibility.reason
                )
            self.custom_cuda_transform = CudaStandardViTTransform(
                image_processor,
                self.input_size,
                self.device,
                vision_adapter=vision_adapter,
            )
            print(
                "[sim-eval] custom ViT preprocessing: CUDA "
                f"({type(image_processor).__name__})"
            )
        self.internvl_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.internvl_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.frames: deque[np.ndarray] = deque(maxlen=self.history_capacity)
        self.history_box_cache = HistoryBoxCache(self.history_capacity)
        self.mask_probabilities: deque[Optional[np.ndarray]] = deque(
            maxlen=self.history_capacity
        )
        self.last_command = np.zeros(3, dtype=np.float32)
        self.last_prediction: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_mask_probability: Optional[np.ndarray] = None
        self.last_grounding_bbox: Optional[List[float]] = None
        self.last_grounding_visibility: Optional[float] = None
        self.last_vqa: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.frames.clear()
        self.history_box_cache.clear()
        self.mask_probabilities.clear()
        self.last_command = np.zeros(3, dtype=np.float32)
        self.last_prediction = None
        self.last_mask = None
        self.last_mask_probability = None
        self.last_grounding_bbox = None
        self.last_grounding_visibility = None
        self.last_vqa = {}

    @staticmethod
    def _to_rgb_array(observation: Any) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(
                "Expected an HxWx3 (or HxWx4) RGB Habitat observation, got "
                f"shape {array.shape}"
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(
                "Expected a non-empty RGB Habitat observation, got "
                f"shape {array.shape}"
            )
        array = array[..., :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
                if array.max() <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        # Detach the RGB array from Habitat's observation buffer before the
        # simulator renders the next frame or reconfigures a scene.
        return np.array(array, dtype=np.uint8, copy=True, order="C")

    @staticmethod
    def _to_rgb_image(observation: Any) -> Image.Image:
        return Image.fromarray(
            TrackVLAHabitatAgent._to_rgb_array(observation), mode="RGB"
        ).copy()

    def observe(self, rgb_observation: Any) -> np.ndarray:
        self.frames.append(self._to_rgb_array(rgb_observation))
        # The current frame has no prediction until this step's single model
        # forward finishes. It remains invalid when inference is skipped.
        self.mask_probabilities.append(None)
        self.history_box_cache.observe()
        return self.frames[-1]

    def _history_slots(self) -> List[Tuple[int, bool]]:
        if not self.frames:
            raise RuntimeError("observe() must be called before predict()")
        newest = len(self.frames) - 1
        indices = [
            newest - offset * self.history_stride
            for offset in range(self.history_size - 1, -1, -1)
            if newest - offset * self.history_stride >= 0
        ]
        # Match the offline loader: when the requested past index predates the
        # episode, repeat the earliest actually sampled RGB frame on the left,
        # but keep those synthetic slots explicitly invalid for target masks.
        padding = self.history_size - len(indices)
        slots = [(indices[0], False)] * padding
        slots.extend((index, True) for index in indices)
        return slots[-self.history_size :]

    def _history_indices(self) -> List[int]:
        return [index for index, _ in self._history_slots()]

    def _history_samples(
        self,
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        slots = self._history_slots()
        frames = list(self.frames)
        masks = list(self.mask_probabilities)
        samples: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        for index, is_real_slot in slots:
            mask = masks[index] if is_real_slot else None
            samples.append(
                (
                    frames[index].copy(),
                    None if mask is None else mask.copy(),
                )
            )
        return samples

    def _history_frames(self) -> List[np.ndarray]:
        return [frame for frame, _ in self._history_samples()]

    def _effective_prompt(self, instruction: Optional[str]) -> str:
        instruction = (instruction or self.default_instruction).strip()
        return "\n".join(part for part in (instruction, self.prompt) if part)

    def _internvl_resize_pad_geometry(
        self,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int, int, int]:
        """Return the shared longest-side resize and padding geometry."""
        return resize_longest_side_and_pad_geometry(
            height,
            width,
            self.input_size,
        )

    def _internvl_pixel_values(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Prepare InternVL RGB tensors while keeping the CPU fallback exact."""
        if self.device.type != "cuda":
            return torch.stack(
                [
                    self.cpu_transform(Image.fromarray(frame, mode="RGB"))
                    for frame in frames
                ],
                dim=0,
            ).to(self.device)

        pixel_values = torch.from_numpy(np.stack(frames, axis=0)).to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = pixel_values.shape
        resized_height, resized_width, top, bottom, left, right = (
            self._internvl_resize_pad_geometry(height, width)
        )
        if (height, width) != (resized_height, resized_width):
            pixel_values = F.interpolate(
                pixel_values,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        pixel_values = F.pad(
            pixel_values,
            (left, right, top, bottom),
            mode="constant",
            value=0.0,
        ).clamp_(0.0, 255.0)
        return pixel_values.div_(255.0).sub_(self.internvl_mean).div_(self.internvl_std)

    def _internvl_batch(
        self, frames: Sequence[np.ndarray], prompt: str
    ) -> Dict[str, torch.Tensor]:
        pixel_values = self._internvl_pixel_values(frames)
        token_counts = self.model_extensions.token_layout(
            len(frames),
            self.num_image_tokens,
        ).per_frame_counts
        image_tokens = "\n".join(
            "<img>" + "<IMG_CONTEXT>" * token_count + "</img>"
            for token_count in token_counts
        )
        text = f"{image_tokens}\nUser: {prompt}\nAssistant:"
        tokenized = self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"].to(self.device),
            "attention_mask": tokenized["attention_mask"].to(self.device),
        }

    def _internvl_history_mask_batch(
        self,
        samples: Optional[
            Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform cached past-mask probabilities into InternVL coordinates."""

        if samples is None:
            samples = self._history_samples()
        past_samples = list(samples[:-1])
        if not past_samples:
            return (
                torch.zeros(
                    (1, 0, self.input_size, self.input_size),
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.zeros((1, 0), device=self.device, dtype=torch.bool),
            )

        transformed_masks: List[torch.Tensor] = []
        mask_valid: List[bool] = []
        for frame, probability in past_samples:
            height, width = frame.shape[:2]
            if probability is None:
                transformed_masks.append(
                    torch.zeros(
                        (self.input_size, self.input_size),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                mask_valid.append(False)
                continue
            if probability.ndim != 2 or probability.shape != (height, width):
                raise ValueError(
                    "Cached mask probability must match its RGB frame; got "
                    f"mask {probability.shape} and RGB {(height, width)}"
                )

            mask_tensor = torch.from_numpy(
                np.array(probability, dtype=np.float32, copy=True, order="C")
            ).to(self.device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            resized_height, resized_width, top, bottom, left, right = (
                self._internvl_resize_pad_geometry(height, width)
            )
            if (height, width) != (resized_height, resized_width):
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            mask_tensor = F.pad(
                mask_tensor,
                (left, right, top, bottom),
                mode="constant",
                value=0.0,
            )
            transformed_masks.append(mask_tensor[0, 0].clamp_(0.0, 1.0))
            mask_valid.append(True)

        return (
            torch.stack(transformed_masks, dim=0).unsqueeze(0),
            torch.tensor(
                [mask_valid],
                device=self.device,
                dtype=torch.bool,
            ),
        )

    def _qwen2vl_batch(
        self, frames: Sequence[np.ndarray], prompt: str
    ) -> Dict[str, torch.Tensor]:
        processor = self.model.tokenizer
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=list(images),
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
            "pixel_values": inputs["pixel_values"].to(self.device),
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "image_grid_thw": inputs["image_grid_thw"].to(self.device),
            "mm_token_type_ids": inputs.get(
                "mm_token_type_ids", torch.zeros_like(inputs["input_ids"])
            ).to(self.device),
        }

    def _custom_batch(
        self,
        frames: Sequence[np.ndarray],
        prompt: str,
    ) -> Dict[str, torch.Tensor]:
        if self.custom_cuda_transform is not None:
            pixel_values = self.custom_cuda_transform(frames)
        else:
            images = [Image.fromarray(frame, mode="RGB") for frame in frames]
            pixel_values = preprocess_custom_images(
                self.model.image_processor,
                images,
                self.input_size,
            )
        token_counts = self.model_extensions.token_layout(
            len(frames),
            self.num_image_tokens,
        ).per_frame_counts
        text = build_custom_multimodal_prompt(
            self.model.tokenizer,
            self.model.image_token,
            token_counts,
            prompt,
        )
        tokenized = self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(self.max_length, sum(token_counts) + 256),
        )
        return {
            "pixel_values": pixel_values.to(self.device),
            "input_ids": tokenized["input_ids"].to(self.device),
            "attention_mask": tokenized["attention_mask"].to(self.device),
        }

    def _qwen3vl_batch(
        self, frames: Sequence[np.ndarray], prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Build a fixed-square Qwen3-VL image batch for stable compression."""
        processor = self.model.tokenizer
        square_transform = ResizeLongestSideAndPad(self.input_size)
        images = [
            square_transform(Image.fromarray(frame, mode="RGB"))
            for frame in frames
        ]
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": prompt})
        text = apply_qwen_vl_chat_template(
            processor,
            [{"role": "user", "content": content}],
            model_type="qwen3vl",
            add_generation_prompt=True,
        )
        fixed_pixels = self.input_size * self.input_size
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(self.max_length, 4096),
            size={
                "shortest_edge": fixed_pixels,
                "longest_edge": fixed_pixels,
            },
        )
        return {
            "pixel_values": inputs["pixel_values"].to(self.device),
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "image_grid_thw": inputs["image_grid_thw"].to(self.device),
        }

    def _set_predicted_mask(
        self,
        mask_logits: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> None:
        """Store a thresholded mask in the current Habitat RGB coordinate space."""
        if mask_logits.ndim == 4:
            if mask_logits.shape[0] != 1 or mask_logits.shape[1] != 1:
                raise ValueError(
                    "Habitat evaluation expects one SAM2 mask logit map per sample, got "
                    f"{tuple(mask_logits.shape)}"
                )
            mask_logits = mask_logits[0, 0]
        elif mask_logits.ndim == 3:
            if mask_logits.shape[0] != 1:
                raise ValueError(
                    "Habitat evaluation expects one SAM2 mask channel, got "
                    f"{tuple(mask_logits.shape)}"
                )
            mask_logits = mask_logits[0]
        if mask_logits.ndim != 2:
            raise ValueError(
                "SAM2 mask logits must be a two-dimensional map, got "
                f"{tuple(mask_logits.shape)}"
            )
        probabilities = torch.sigmoid(mask_logits.float()).unsqueeze(0).unsqueeze(0)
        if self.model_type in {"custom", "internvl", "internvl3", "qwen3vl"}:
            probabilities = self._restore_internvl_mask_geometry(
                probabilities,
                output_size,
            )
        else:
            width, height = output_size
            probabilities = torch.nn.functional.interpolate(
                probabilities,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        probabilities = probabilities[0, 0]
        self.last_mask_probability = probabilities.detach().cpu().numpy()
        self.last_mask = (
            (probabilities >= 0.5).to(dtype=torch.uint8).mul(255).detach().cpu().numpy()
        )

    def _restore_internvl_mask_geometry(
        self,
        probabilities: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Remove padding and invert the longest-side resize used by InternVL."""
        output_width, output_height = output_size
        model_mask = torch.nn.functional.interpolate(
            probabilities,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        resized_height, resized_width, top, _, left, _ = (
            self._internvl_resize_pad_geometry(output_height, output_width)
        )
        unpadded_mask = model_mask[
            :,
            :,
            top : top + resized_height,
            left : left + resized_width,
        ]
        return torch.nn.functional.interpolate(
            unpadded_mask,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )

    def predict(self, instruction: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Auxiliary outputs belong to the current inference call. Clear them
        # first so a checkpoint without a given head cannot leak stale values
        # from the previous frame into the next *_info.json record.
        self.last_grounding_bbox = None
        self.last_grounding_visibility = None
        self.last_vqa = {}
        samples = self._history_samples()
        frames = [frame for frame, _ in samples]
        prompt = self._effective_prompt(instruction)
        if self.model_type in {"internvl", "internvl3"}:
            batch = self._internvl_batch(frames, prompt)
            if self.extension_requirements.online_mask_cache:
                history_masks, history_mask_valid = (
                    self._internvl_history_mask_batch(samples)
                )
                batch["history_masks"] = history_masks
                batch["history_mask_valid"] = history_mask_valid
        elif self.model_type == "custom":
            batch = self._custom_batch(frames, prompt)
            if self.extension_requirements.online_mask_cache:
                history_masks, history_mask_valid = (
                    self._internvl_history_mask_batch(samples)
                )
                batch["history_masks"] = history_masks
                batch["history_mask_valid"] = history_mask_valid
        elif self.model_type == "qwen2vl":
            batch = self._qwen2vl_batch(frames, prompt)
        elif self.model_type == "qwen3vl":
            batch = self._qwen3vl_batch(frames, prompt)
            if self.extension_requirements.online_mask_cache:
                history_masks, history_mask_valid = (
                    self._internvl_history_mask_batch(samples)
                )
                batch["history_masks"] = history_masks
                batch["history_mask_valid"] = history_mask_valid
        else:
            raise RuntimeError(f"Unsupported model type: {self.model_type}")

        if self.extension_requirements.online_box_cache:
            boxes, valid = self.history_box_cache.batch(
                self._history_slots(),
                threshold=self.model_extensions.history_compressor.bbox_visibility_threshold,
                device=self.device)
            batch["history_boxes"] = boxes
            batch["history_box_valid"] = valid

        autocast = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype in (torch.bfloat16, torch.float16)
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            model_outputs = self.model(**batch)

        # Grounding head emits normalized cx, cy, width, height in [0, 1].
        grounding_bbox = model_outputs.get("grounding_bbox")
        if grounding_bbox is not None:
            values = grounding_bbox[0].detach().float().cpu().reshape(-1).tolist()
            if len(values) >= 4:
                self.last_grounding_bbox = [float(value) for value in values[:4]]
        visibility_logits = model_outputs.get("grounding_visibility_logits")
        if visibility_logits is not None:
            self.last_grounding_visibility = float(
                torch.sigmoid(visibility_logits[0].detach().float()).reshape(-1)[0].item()
            )

        if self.extension_requirements.online_box_cache:
            if grounding_bbox is None or visibility_logits is None:
                raise RuntimeError("bbox target_selection requires grounding_bbox and grounding_visibility_logits outputs")
            self.history_box_cache.update(self.last_grounding_bbox, self.last_grounding_visibility)
            if self.last_grounding_bbox is not None:
                # Cache stays in padded model space; overlays/JSON retain the
                # existing original-RGB normalized coordinate convention.
                height, width = frames[-1].shape[:2]
                box = torch.tensor(self.last_grounding_bbox, dtype=torch.float32)
                geometry = self._internvl_resize_pad_geometry(height, width)
                self.last_grounding_bbox = resize_pad_boxes(
                    box, geometry, self.input_size, inverse=True).clamp(0, 1).tolist()

        # VQA heads emit logits for position/scale/distance/facing. Save both
        # the class id and readable label, together with confidence.
        for task_name, class_names in VQA_CLASS_NAMES.items():
            logits = model_outputs.get(f"vqa_{task_name}_logits")
            if logits is None:
                continue
            probabilities = torch.softmax(logits[0].detach().float(), dim=-1).reshape(-1)
            class_id = int(probabilities.argmax().item())
            self.last_vqa[task_name] = {
                "class_id": class_id,
                "class_name": class_names[class_id]
                if class_id < len(class_names)
                else str(class_id),
                "confidence": float(probabilities[class_id].item()),
            }
        prediction = model_outputs["pred_traj"]
        mask_logits = model_outputs.get("mask_logits")
        if mask_logits is None:
            self.last_mask = None
            self.last_mask_probability = None
            self.mask_probabilities[-1] = None
        else:
            height, width = frames[-1].shape[:2]
            self._set_predicted_mask(mask_logits, (width, height))
            if self.last_mask_probability is None:
                raise RuntimeError("Mask postprocessing did not produce a probability map")
            if self.extension_requirements.online_mask_cache:
                # Write the current prediction back into the same record that
                # was appended by observe(). Future QFormer steps consume it.
                self.mask_probabilities[-1] = self.last_mask_probability.copy()
            else:
                # Adaptive pooling and the legacy dense path never retain
                # historical masks.
                self.mask_probabilities[-1] = None
        trajectory = prediction[0].detach().float().cpu().numpy()
        if self.waypoint_index >= trajectory.shape[0]:
            raise IndexError(
                f"waypoint_index={self.waypoint_index} is outside prediction with "
                f"{trajectory.shape[0]} points"
            )
        command = np.zeros(3, dtype=np.float32)
        point = trajectory[self.waypoint_index]
        command[: min(3, point.shape[0])] = point[:3]
        command /= self.control_dt
        if self.clip_commands:
            command = np.clip(command, -1.0, 1.0)
        self.last_command = command.astype(np.float32, copy=False)
        self.last_prediction = trajectory
        return self.last_command, trajectory


def is_trackopen_code_root(path: Path) -> bool:
    return (path / "habitat-lab").is_dir() and (path / "evt_bench").is_dir()


def asset_parent(path: Path) -> Optional[Path]:
    """Return the directory that contains a TrackOpen-style ``data/`` tree."""
    if (path / "data").is_dir():
        return path
    if path.name == "data" and path.is_dir():
        return path.parent
    return None


def resolve_trackvla_paths(
    trackvla_root_arg: str,
    code_root_arg: Optional[str],
    assets_root_arg: Optional[str],
) -> Tuple[Path, Path]:
    """Resolve separate TrackOpen code and TrackVLA asset roots."""
    requested_root = resolve_existing_path(trackvla_root_arg, "TrackVLA root")
    if not requested_root.is_dir():
        raise NotADirectoryError(f"TrackVLA root is not a directory: {requested_root}")

    code_candidates: List[Path] = []
    for value in (code_root_arg, os.environ.get("TRACKVLA_CODE_ROOT")):
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            code_candidates.append(candidate)
    if is_trackopen_code_root(requested_root):
        code_candidates.append(requested_root)
    # These cover a checkout copied into this repository and the usual sibling
    # layout: /path/to/trackopen next to /path/to/embodvla.
    code_candidates.extend((PROJECT_ROOT / "trackopen", PROJECT_ROOT.parent / "trackopen"))

    code_root = next(
        (candidate.resolve() for candidate in code_candidates if is_trackopen_code_root(candidate)),
        None,
    )
    if code_root is None:
        raise FileNotFoundError(
            "Could not find a TrackOpen checkout with habitat-lab/ and evt_bench/. "
            "Pass --trackvla-code-root /path/to/trackopen when --trackvla-root "
            "points at an asset pack."
        )

    if assets_root_arg:
        asset_candidate = resolve_existing_path(assets_root_arg, "TrackOpen assets root")
    elif not is_trackopen_code_root(requested_root):
        asset_candidate = requested_root
    else:
        local_asset_pack = PROJECT_ROOT / "trackvla_bench"
        asset_candidate = local_asset_pack if (local_asset_pack / "data").is_dir() else code_root

    assets_root = asset_parent(asset_candidate)
    if assets_root is None:
        raise FileNotFoundError(
            f"TrackOpen assets root must contain data/: {asset_candidate}. "
            "Pass --trackvla-assets-root /path/to/asset-pack."
        )

    required_paths = (
        assets_root / "data" / "datasets" / "track",
        assets_root / "data" / "humanoids",
        assets_root / "data" / "robots" / "hab_spot_arm",
        assets_root / "data" / "scene_datasets",
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "TrackOpen asset pack is incomplete; missing: " + ", ".join(missing_paths)
        )
    return code_root, assets_root


@contextlib.contextmanager
def trackopen_asset_workdir(trackopen_code_root: Path, trackopen_assets_root: Path):
    """Expose local assets and TrackOpen's root-level runtime files together."""
    if trackopen_code_root == trackopen_assets_root:
        previous_workdir = Path.cwd()
        os.chdir(trackopen_assets_root)
        try:
            yield trackopen_assets_root
        finally:
            os.chdir(previous_workdir)
        return

    runtime_files = ("humanoid_infos.json",)
    missing_files = [
        str(trackopen_code_root / filename)
        for filename in runtime_files
        if not (trackopen_code_root / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "TrackOpen runtime is missing required files: " + ", ".join(missing_files)
        )

    with tempfile.TemporaryDirectory(prefix="trackvla_habitat_") as temporary_dir:
        workdir = Path(temporary_dir)
        os.symlink(
            trackopen_assets_root / "data",
            workdir / "data",
            target_is_directory=True,
        )
        for filename in runtime_files:
            os.symlink(trackopen_code_root / filename, workdir / filename)

        previous_workdir = Path.cwd()
        os.chdir(workdir)
        try:
            yield workdir
        finally:
            os.chdir(previous_workdir)


def add_trackopen_imports(trackopen_code_root: Path) -> None:
    for path in (trackopen_code_root / "habitat-lab", trackopen_code_root, PROJECT_ROOT):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def import_trackopen_runtime(trackopen_code_root: Path):
    add_trackopen_imports(trackopen_code_root)
    try:
        import habitat
        from habitat.datasets import make_dataset
        import evt_bench  # noqa: F401 - registers TrackOpen custom components.
        from habitat_sim.gfx import LightInfo, LightPositionModel
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise RuntimeError(
            "TrackOpen Habitat dependencies are unavailable. Run this script in an "
            "environment containing TrackOpen's habitat_sim (normally conda env evt_bench)."
        ) from exc
    return habitat, make_dataset, LightInfo, LightPositionModel, GlobalHydra


def clear_initialized_global_hydra(global_hydra: Any) -> None:
    """Release a prior Hydra composition before Habitat builds its config."""
    instance = global_hydra.instance()
    if instance.is_initialized():
        instance.clear()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_instruction(episode: Any, default_instruction: str) -> str:
    info = getattr(episode, "info", None)
    if info is not None:
        try:
            instruction = info.get("instruction")
        except AttributeError:
            instruction = None
        if instruction:
            return str(instruction)
    return default_instruction


def robot_target_distance(sim: Any) -> Optional[float]:
    try:
        target = np.asarray(sim.agents_mgr[0].articulated_agent.base_pos, dtype=np.float32)
        robot = np.asarray(sim.agents_mgr[1].articulated_agent.base_pos, dtype=np.float32)
    except (AttributeError, IndexError, TypeError):
        return None
    return float(np.linalg.norm(robot - target))


def apply_reference_light_setup(sim: Any, light_info: Any, light_position_model: Any) -> None:
    try:
        sim.set_light_setup(
            [
                light_info(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=light_position_model.Global),
                light_info(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=light_position_model.Global),
                light_info(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=light_position_model.Global),
                light_info(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=light_position_model.Global),
            ]
        )
    except Exception as exc:
        print(f"[sim-eval] Could not apply reference light setup: {exc}")


def overlay_predicted_mask(
    image: Image.Image,
    mask: Optional[np.ndarray],
) -> Image.Image:
    """Highlight a predicted binary mask with a shallow-yellow video overlay."""
    if mask is None:
        return image

    mask_array = np.asarray(mask)
    if mask_array.shape != (image.height, image.width):
        return image

    mask_alpha = Image.fromarray(
        np.where(mask_array > 0, 92, 0).astype(np.uint8),
        mode="L",
    )
    overlay = Image.new("RGBA", image.size, (255, 244, 153, 0))
    overlay.putalpha(mask_alpha)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_predicted_trajectory(
    image: Image.Image,
    trajectory: Optional[np.ndarray],
) -> np.ndarray:
    """Add a top-down ego-frame trajectory inset to a rendered RGB frame."""

    width, height = image.size
    margin = max(6, min(width, height) // 48)
    panel_size = min(max(128, min(width, height) // 3), min(width, height) - 2 * margin)
    panel_size = max(32, panel_size)
    left = width - margin - panel_size
    top = height - margin - panel_size
    right = left + panel_size
    bottom = top + panel_size

    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (left, top, right, bottom),
        fill=(8, 14, 20, 210),
        outline=(220, 230, 235, 220),
        width=1,
    )

    inset_margin = max(10, panel_size // 12)
    inner_left = left + inset_margin
    inner_top = top + inset_margin + 10
    inner_right = right - inset_margin
    inner_bottom = bottom - inset_margin
    lateral_limit = 5.0
    forward_min = -1.0
    forward_max = 6.0
    origin_x = (inner_left + inner_right) // 2
    origin_y = int(
        inner_top
        + (forward_max / (forward_max - forward_min)) * (inner_bottom - inner_top)
    )

    draw.text((left + inset_margin, top + 2), "PRED TRAJ", fill=(230, 240, 245, 255))
    draw.line((origin_x, inner_top, origin_x, inner_bottom), fill=(110, 125, 135, 180), width=1)
    draw.line((inner_left, origin_y, inner_right, origin_y), fill=(110, 125, 135, 180), width=1)

    def to_panel_point(forward: float, lateral: float) -> Tuple[int, int]:
        pixel_x = origin_x - lateral / lateral_limit * (inner_right - inner_left) / 2.0
        pixel_y = inner_top + (forward_max - forward) / (forward_max - forward_min) * (
            inner_bottom - inner_top
        )
        return (
            int(np.clip(pixel_x, inner_left, inner_right)),
            int(np.clip(pixel_y, inner_top, inner_bottom)),
        )

    origin_radius = max(3, panel_size // 35)
    draw.ellipse(
        (
            origin_x - origin_radius,
            origin_y - origin_radius,
            origin_x + origin_radius,
            origin_y + origin_radius,
        ),
        fill=(255, 90, 80, 255),
    )

    if trajectory is not None:
        trajectory_array = np.asarray(trajectory, dtype=np.float32)
        if trajectory_array.ndim == 2 and trajectory_array.shape[1] >= 2:
            finite = np.isfinite(trajectory_array[:, :2]).all(axis=1)
            points = [
                to_panel_point(float(point[0]), float(point[1]))
                for point in trajectory_array[finite]
            ]
            if len(points) >= 2:
                draw.line(points, fill=(0, 235, 255, 255), width=max(2, panel_size // 55))
            point_radius = max(2, panel_size // 50)
            for point_x, point_y in points:
                draw.ellipse(
                    (
                        point_x - point_radius,
                        point_y - point_radius,
                        point_x + point_radius,
                        point_y + point_radius,
                    ),
                    fill=(100, 255, 170, 255),
                )

    return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB")).copy()


def draw_auxiliary_predictions(
    image: Image.Image,
    bbox: Optional[Sequence[float]],
    vqa: Optional[Mapping[str, Mapping[str, Any]]],
) -> Image.Image:
    """Overlay normalized grounding box and readable VQA predictions."""
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    if bbox is not None and len(bbox) >= 4:
        cx, cy, bw, bh = [float(value) for value in bbox[:4]]
        width, height = canvas.size
        x0 = max(0, min(width - 1, int((cx - bw / 2.0) * width)))
        y0 = max(0, min(height - 1, int((cy - bh / 2.0) * height)))
        x1 = max(0, min(width - 1, int((cx + bw / 2.0) * width)))
        y1 = max(0, min(height - 1, int((cy + bh / 2.0) * height)))
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64, 255), width=3)
        draw.text((x0, max(0, y0 - 14)), "grounding", fill=(255, 64, 64, 255))
    if vqa:
        lines = [
            f"{task}: {value.get('class_name', value.get('class_id'))} "
            f"({float(value.get('confidence', 0.0)):.2f})"
            for task, value in vqa.items()
        ]
        draw.multiline_text(
            (8, 8), "\n".join(lines), fill=(255, 255, 255, 255),
            stroke_width=2, stroke_fill=(0, 0, 0, 220),
        )
    return canvas.convert("RGB")


def output_paths(save_path: Path, episode: Any) -> Tuple[Path, str]:
    scene_id = str(getattr(episode, "scene_id", "unknown_scene"))
    scene_key = Path(scene_id).name.split(".")[0] or "unknown_scene"
    episode_id = str(getattr(episode, "episode_id", "unknown_episode")).replace("/", "_")
    directory = save_path / scene_key
    directory.mkdir(parents=True, exist_ok=True)
    return directory, episode_id


def evaluate_split(
    habitat: Any,
    habitat_config: Any,
    dataset_split: Any,
    agent: TrackVLAHabitatAgent,
    args: argparse.Namespace,
    light_info: Any,
    light_position_model: Any,
    save_path: Path,
) -> Dict[str, Any]:
    action_names = tuple(part.strip() for part in args.action_names.split(",") if part.strip())
    if not action_names:
        raise ValueError("--action-names must include at least one action")
    if args.robot_action not in action_names:
        raise ValueError("--robot-action must be included in --action-names")

    images_to_video = None
    if args.save_video:
        from habitat.utils.visualizations.utils import images_to_video as habitat_images_to_video

        images_to_video = habitat_images_to_video

    all_results: List[Dict[str, Any]] = []
    with habitat.TrackEnv(config=habitat_config, dataset=dataset_split) as env:
        sim = env.sim
        for episode_index in range(len(env.episodes)):
            # Drop the prior episode's RGB history before Habitat potentially
            # reconfigures the native simulator for a new scene.
            agent.reset()
            observations = env.reset()
            apply_reference_light_setup(sim, light_info, light_position_model)
            instruction = episode_instruction(env.current_episode, args.default_instruction)
            records: List[Dict[str, Any]] = []
            status = "Normal"
            too_far_steps = 0
            steps = 0
            followed_steps = 0
            action = np.zeros(3, dtype=np.float32)
            save_episode_video = args.save_video
            video_frames: List[np.ndarray] = []

            while not env.episode_over:
                if args.rgb_sensor not in observations:
                    raise KeyError(
                        f"RGB sensor {args.rgb_sensor!r} is missing. Available keys: "
                        f"{sorted(observations)}"
                    )
                rgb_frame = agent.observe(observations[args.rgb_sensor])
                model_updated = steps % args.inference_interval == 0
                if model_updated:
                    action, _ = agent.predict(instruction)
                else:
                    action = agent.last_command
                if save_episode_video:
                    video_image = draw_auxiliary_predictions(
                        Image.fromarray(rgb_frame, mode="RGB"),
                        agent.last_grounding_bbox,
                        agent.last_vqa,
                    )
                    video_image = overlay_predicted_mask(
                        video_image,
                        agent.last_mask if args.show_masks else None,
                    )
                    video_frames.append(
                        draw_predicted_trajectory(video_image, agent.last_prediction)
                    )
                # TrackEnv.step() already advances the simulator and renders
                # the next observation. Reusing it avoids a second native
                # sensor-render pass for every evaluation step.
                observations = env.step(
                    {
                        "action": action_names,
                        "action_args": {args.robot_action_arg: np.asarray(action, dtype=np.float32)},
                    }
                )
                steps += 1
                metrics = env.get_metrics()
                following = float(metrics.get("human_following", 0.0))
                collision = float(metrics.get("human_collision", 0.0))
                if following >= 1.0:
                    followed_steps += 1
                    too_far_steps = 0

                distance = robot_target_distance(sim)
                if distance is not None and distance > args.max_distance:
                    too_far_steps += 1
                    if too_far_steps > args.max_lost_steps:
                        status = "Lost"
                        break
                elif following < 1.0:
                    too_far_steps = 0

                records.append(
                    {
                        "step": steps,
                        "dis_to_human": distance,
                        "facing": following,
                        "base_velocity": [float(value) for value in action],
                        "model_updated": model_updated,
                        "predicted_trajectory": (
                            agent.last_prediction.tolist()
                            if agent.last_prediction is not None
                            else None
                        ),
                        # Normalized cx, cy, w, h from the grounding head.
                        "grounding_bbox": agent.last_grounding_bbox,
                        "grounding_visibility": agent.last_grounding_visibility,
                        # Dict is copied because the agent reuses this object
                        # on the next inference step.
                        "vqa": dict(agent.last_vqa) if agent.last_vqa else None,
                    }
                )
                if collision >= 1.0 and not args.continue_on_collision:
                    status = "Collision"
                    break
                if args.log_every > 0 and steps % args.log_every == 0:
                    print(
                        f"[sim-eval] episode={env.current_episode.episode_id} step={steps} "
                        f"action={np.array2string(action, precision=3)} "
                        f"distance={distance} following={following:.1f}"
                    )

            if save_episode_video and args.rgb_sensor in observations:
                final_image = TrackVLAHabitatAgent._to_rgb_image(observations[args.rgb_sensor])
                final_image = draw_auxiliary_predictions(
                    final_image, agent.last_grounding_bbox, agent.last_vqa
                )
                video_frames.append(
                    draw_predicted_trajectory(final_image, agent.last_prediction)
                )

            metrics = env.get_metrics()
            finished = bool(env.episode_over)
            following = float(metrics.get("human_following", 0.0))
            success_measure = float(metrics.get("human_following_success", 0.0))
            result: Dict[str, Any] = {
                "finish": finished,
                "status": status,
                "success": bool(success_measure and following) if steps < 300 else bool(following),
                "following_rate": followed_steps / steps if steps else 0.0,
                "following_step": followed_steps,
                "total_step": steps,
                "collision": float(metrics.get("human_collision", 0.0)),
                "instruction": instruction,
                "grounding_bbox_last": agent.last_grounding_bbox,
                "grounding_visibility_last": agent.last_grounding_visibility,
                "vqa_last": dict(agent.last_vqa) if agent.last_vqa else None,
                "has_grounding_predictions": any(
                    record.get("grounding_bbox") is not None for record in records
                ),
                "has_vqa_predictions": any(
                    record.get("vqa") is not None for record in records
                ),
            }
            directory, episode_id = output_paths(save_path, env.current_episode)
            if save_episode_video and video_frames:
                assert images_to_video is not None
                images_to_video(
                    video_frames,
                    output_dir=str(directory),
                    video_name=episode_id,
                    fps=args.video_fps,
                    verbose=False,
                )
                result["video"] = f"{episode_id}.mp4"
            with (directory / f"{episode_id}_info.json").open("w") as handle:
                json.dump(records, handle, indent=2)
            with (directory / f"{episode_id}.json").open("w") as handle:
                json.dump(result, handle, indent=2)
            all_results.append(result)
            print(
                f"[sim-eval] completed {episode_index + 1}/{len(env.episodes)} "
                f"episode={episode_id} status={status} success={result['success']}"
            )
            video_frames.clear()

    count = len(all_results)
    return {
        "split_id": args.split_id,
        "split_num": args.split_num,
        "num_episodes": count,
        "success_rate": sum(bool(result["success"]) for result in all_results) / count if count else 0.0,
        "following_rate": sum(float(result["following_rate"]) for result in all_results) / count if count else 0.0,
        "collision_rate": sum(float(result["collision"]) > 0.0 for result in all_results) / count if count else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.split_num <= 0:
        raise ValueError("split_num must be positive")
    if not 0 <= args.split_id < args.split_num:
        raise ValueError("split_id must be in [0, split_num)")
    if args.inference_interval <= 0:
        raise ValueError("inference_interval must be positive")
    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive")

    checkpoint_path = resolve_existing_path(args.checkpoint, "Checkpoint")
    if args.config is not None:
        config_path = resolve_existing_path(args.config, "VLA config")
    else:
        config_path = checkpoint_config_path(checkpoint_path)
    config = load_checkpoint_config(
        checkpoint_path,
        config_path if args.config is not None else None,
    )
    config_source = str(config_path)
    print(f"[sim-eval] Using VLA config from {config_source}")
    trackopen_code_root, trackopen_assets_root = resolve_trackvla_paths(
        args.trackvla_root,
        args.trackvla_code_root,
        args.trackvla_assets_root,
    )
    exp_config = Path(args.exp_config).expanduser()
    if not exp_config.is_absolute():
        exp_config = trackopen_code_root / exp_config
    exp_config = resolve_existing_path(str(exp_config), "Habitat experiment config")
    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    print(f"[sim-eval] Loading checkpoint on {device}: {checkpoint_path}")
    model, model_type, dtype = load_vla_model(
        config=config,
        checkpoint_path=checkpoint_path,
        model_path_override=args.model_path,
        vision_model_path_override=args.vision_model_path,
        llm_model_path_override=args.llm_model_path,
        device=device,
        allow_hf_download=args.allow_hf_download,
    )
    agent = TrackVLAHabitatAgent(
        model=model,
        model_type=model_type,
        data_config=config["data"],
        device=device,
        dtype=dtype,
        control_dt=args.control_dt,
        waypoint_index=args.waypoint_index,
        clip_commands=not args.no_command_clip,
        default_instruction=args.default_instruction,
    )
    habitat, make_dataset, light_info, light_position_model, global_hydra = import_trackopen_runtime(
        trackopen_code_root
    )
    with trackopen_asset_workdir(trackopen_code_root, trackopen_assets_root):
        clear_initialized_global_hydra(global_hydra)
        habitat_config = habitat.get_config(str(exp_config), args.opts)
        seed = int(habitat_config.habitat.simulator.seed)
        set_seed(seed)
        print(
            f"[sim-eval] split={args.split_id}/{args.split_num}, seed={seed}, "
            f"model_type={model_type}, runtime={trackopen_code_root}, "
            f"assets={trackopen_assets_root}, output={save_path}"
        )

        dataset = make_dataset(
            id_dataset=habitat_config.habitat.dataset.type,
            config=habitat_config.habitat.dataset,
        )
        try:
            splits = dataset.get_splits(
                args.split_num, allow_uneven_splits=args.allow_uneven_splits
            )
        except TypeError:
            splits = dataset.get_splits(args.split_num)
        if args.split_id >= len(splits):
            raise IndexError(
                f"split_id={args.split_id} but dataset produced only {len(splits)} splits"
            )
        summary = evaluate_split(
            habitat=habitat,
            habitat_config=habitat_config,
            dataset_split=splits[args.split_id],
            agent=agent,
            args=args,
            light_info=light_info,
            light_position_model=light_position_model,
            save_path=save_path,
        )
    summary_path = save_path / f"summary_split_{args.split_id:03d}_of_{args.split_num:03d}.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[sim-eval] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
