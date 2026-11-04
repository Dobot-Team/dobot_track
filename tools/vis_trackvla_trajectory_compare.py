#!/usr/bin/env python3
"""Render TrackVLA ground-truth and model trajectories for a ``frames.jpgs``.

The tool reads TrackVLA's ZIP image archive and its matching ``samples.jsonl``
manifest, restores a VLA checkpoint, and writes a comparison video.  Every
output frame contains the current camera image, the GT/prediction trajectory,
and useful metadata (frame id, prompt, EPE/ADE, FDE and yaw error).

TrackVLA trajectories are in the ego ground plane (forward ``x``, left ``y``),
whereas Dobot camera calibration is not stored in the JSONL/archive.  Therefore
the default display uses a metrically correct ego-frame inset.  To draw the
same trajectories directly on the camera image, pass a calibrated 3x3 ground
plane homography with ``--homography``.  It must map ``[ego_x, ego_y, 1]`` to
image pixels ``[u, v, w]``.

Example (the archive in the training data):
    python tools/vis_trackvla_trajectory_compare.py ^
      --frames data/dobot-trackvla/frames/dobot_camera_history/20260706/F-01-016-004-1783318837612380539/dobot_bag_0000_0/frames.jpgs ^
      --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml ^
      --checkpoint outputs/<run>/checkpoints/last.ckpt ^
      --output outputs/trackvla_compare.mp4 --max-frames 120

Direct image overlay (H is row-major and maps ego metres to image pixels):
    python tools/vis_trackvla_trajectory_compare.py ... ^
      --homography h00 h01 h02 h10 h11 h12 h20 h21 h22
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from vla.config import ExecutionStage, ModelExtensionContext, ModelExtensionsConfig
from vla.data.trackvla import TrackVLADataset
from vla.lightning import VLAPlModule
from vla.utils.common import build_transform
from vla.utils.checkpoint_config import (
    checkpoint_config_path,
    load_checkpoint_config,
    load_checkpoint_hparams,
)


# OpenCV uses BGR. These deliberately match the reference convention.
GT_COLOR = (0, 220, 255)  # yellow
PRED_COLOR = (255, 255, 0)  # cyan
EGO_COLOR = (0, 220, 0)  # green
TEXT_COLOR = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT vs model TrackVLA trajectories from a frames.jpgs archive."
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="TrackVLA frames.jpgs ZIP archive.",
    )
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional JSONL manifest. It is inferred from --frames when omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning .ckpt file (or a DeepSpeed checkpoint directory).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training YAML. Defaults to config.yaml next to --checkpoint.",
    )
    parser.add_argument("--output", required=True, help="Output .mp4 path.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based record index in the JSONL manifest.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Render at most this many manifest records.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render one record every N records.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, cuda:0, or cpu (default: auto).",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable CUDA mixed-precision inference.",
    )
    parser.add_argument(
        "--homography",
        type=float,
        nargs=9,
        default=None,
        metavar=("H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21", "H22"),
        help="Optional row-major 3x3 homography from ego (x forward, y left) metres to image pixels.",
    )
    parser.add_argument(
        "--cpu-transform",
        action="store_true",
        help="Use the legacy PIL/torchvision CPU image transform even on CUDA.",
    )
    parser.add_argument(
        "--inset-position",
        choices=("bottom-center", "center", "bottom-right"),
        default="bottom-center",
        help="Placement of the ego-frame trajectory inset without --homography (default: bottom-center).",
    )
    parser.add_argument(
        "--trajectory-range",
        type=float,
        nargs=2,
        default=(3.0, 2.0),
        metavar=("FORWARD_M", "LATERAL_M"),
        help="Fixed ego-inset range: forward [0, FORWARD_M] and lateral +/-LATERAL_M metres.",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="Keep source resolution instead of upscaling output frames below 1080p.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Optional TrueType/OpenType font. Pass a CJK font to render Chinese prompts.",
    )
    return parser.parse_args()

def manifest_references_frames(manifest_path: Path, frames_path: Path, subset_root: Path) -> bool:
    """Check that a JSONL manifest belongs to the requested frame archive."""

    expected_frames_file = frames_path.relative_to(subset_root).as_posix()
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frames_file = record.get("frames_file") if isinstance(record, dict) else None
            return (
                isinstance(frames_file, str)
                and frames_file.replace("\\", "/").lstrip("./") == expected_frames_file
            )
    return False


def resolve_samples_path(frames_path: Path, explicit_path: Optional[str]) -> Path:
    """Locate the manifest paired with an archive in old and new TrackVLA layouts."""

    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--samples does not exist: {path}")
        return path

    parts = list(frames_path.parts)
    try:
        frames_index = len(parts) - 1 - parts[::-1].index("frames")
    except ValueError as exc:
        raise ValueError(
            "Cannot infer a manifest because --frames is not below a frames/ directory. "
            "Pass --samples explicitly."
        ) from exc

    subset_root = Path(*parts[:frames_index])
    manifest_dir = Path(*parts[:frames_index], "jsonl", *parts[frames_index + 1 : -1])
    candidates = (
        manifest_dir / f"{frames_path.stem}.jsonl",
        manifest_dir / "samples.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file() and manifest_references_frames(candidate, frames_path, subset_root):
            return candidate

    candidate_list = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No manifest matched --frames. Checked:\n  " + candidate_list
        + "\nPass --samples explicitly if this archive uses a custom mapping."
    )
    return candidate


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested with --device {value}, but CUDA is unavailable")
    return device


def load_model(
    checkpoint: str,
    config_path: Optional[str],
    device: torch.device,
) -> Tuple[VLAPlModule, Dict[str, Any], torch.dtype]:
    """Restore a checkpoint with the same extension resolution as scripts/eval.py."""

    config = load_checkpoint_config(checkpoint, config_path)
    if config.get("data", {}).get("dataset_type") != "trackvla":
        raise ValueError(
            "This tool accepts a TrackVLA frames.jpgs archive, so config.data.dataset_type "
            f"must be 'trackvla', got {config.get('data', {}).get('dataset_type')!r}."
        )

    checkpoint_hparams = load_checkpoint_hparams(checkpoint)
    model_config = config["model"]
    data_config = config["data"]
    model_extensions = ModelExtensionsConfig.resolve_checkpoint(
        model_config=model_config,
        checkpoint_hparams=checkpoint_hparams,
        context=ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type="trackvla",
            model_type=(model_config.get("model_type") or checkpoint_hparams.get("model_type")),
            dense_tokens_per_frame=data_config.get("num_image_tokens", 256),
        ),
    )
    # This may fill legacy extension defaults based on the saved checkpoint.
    model_extensions.apply_to_model_mapping(model_config)
    model_extensions.validate_for(
        ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type="trackvla",
            model_type=(model_config.get("model_type") or checkpoint_hparams.get("model_type")),
            dense_tokens_per_frame=data_config.get("num_image_tokens", 256),
        )
    )

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(model_config.get("dtype", "bf16"), torch.bfloat16)
    module = VLAPlModule.load_from_checkpoint(
        checkpoint,
        model_path=model_config["model_path"],
        model_type=model_config.get("model_type"),
        num_queries=model_config["num_queries"],
        n_future=data_config["n_future"],
        freeze_vision=model_config["freeze_vision"],
        freeze_llm=model_config["freeze_llm"],
        freeze_mlp1=model_config.get("freeze_mlp1", True),
        freeze_action_head=model_config.get("freeze_action_head", False),
        action_head_dims=tuple(model_config["action_head_dims"]),
        action_head_type=model_config.get("action_head_type", "legacy"),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        yaw_weight=config["training"].get("yaw_weight", 1.0),
        x_weight=config["training"].get("x_weight", 1.0),
        y_weight=config["training"].get("y_weight", 1.0),
        dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        sam2_config=model_extensions.sam2,
        history_compressor_config=model_extensions.history_compressor,
        auxiliary_tasks_config=model_config.get("auxiliary_tasks_config"),
    )
    module.to(device)
    module.eval()
    module.freeze()
    return module, config, dtype


def build_dataset(module: VLAPlModule, config: Dict[str, Any], samples_path: Path) -> TrackVLADataset:
    data_config = config["data"]
    model_extensions = ModelExtensionsConfig.from_model_mapping(config["model"])
    return TrackVLADataset(
        source_paths=str(samples_path),
        tokenizer=module.model.tokenizer,
        transform=build_transform(int(data_config["input_size"])),
        num_history_frames=int(data_config["num_history_frames"]),
        history_frame_interval=int(data_config.get("history_frame_interval", 0)),
        n_future=int(data_config["n_future"]),
        trajectory_dim=int(data_config.get("trajectory_dim", 3)),
        prompt=str(data_config.get("prompt", "")),
        num_image_tokens=int(data_config.get("num_image_tokens", 256)),
        max_length=int(data_config.get("max_length", 1200)),
        model_type=module.model_type,
        processor=(
            module.model.tokenizer
            if module.model_type in {"qwen2vl", "qwen3vl"}
            else None
        ),
        qwen_min_pixels=int(data_config.get("qwen_min_pixels", 3136)),
        qwen_max_pixels=int(data_config.get("qwen_max_pixels", 100_000)),
        # A one-episode manifest is cheap to index and this tool never writes a cache.
        index_cache_dir=None,
        max_open_archives=1,
        load_masks=model_extensions.sam2.enabled,
        auxiliary_tasks_config=module.auxiliary_tasks_config,
        history_compressor_config=config["model"].get("history_compressor_config"),
        history_mask_augmentation=False,
        mask_input_size=int(data_config["input_size"]),
    )


def prepare_model_batch(sample: Dict[str, Any], model_type: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert one TrackVLADataset item to the model's inference batch layout."""

    batch: Dict[str, torch.Tensor] = {
        "pixel_values": sample["pixel_values"].to(device),
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }
    if model_type in {"qwen2vl", "qwen3vl"}:
        batch["image_grid_thw"] = sample["image_grid_thw"].to(device)
        batch["mm_token_type_ids"] = sample["mm_token_type_ids"].unsqueeze(0).to(device)
    if "history_masks" in sample:
        batch["history_masks"] = sample["history_masks"].unsqueeze(0).to(device)
        batch["history_mask_valid"] = sample["history_mask_valid"].unsqueeze(0).to(device)
    if "mask" in sample:
        batch["masks"] = sample["mask"].unsqueeze(0).to(device)
    for key in ("history_boxes", "history_box_valid", "grounding_bbox_target", "grounding_visibility_target", "vqa_labels"):
        if key in sample:
            batch[key] = sample[key].unsqueeze(0).to(device)
    return batch


def internvl_cuda_pixel_values(
    images: Sequence[Image.Image],
    input_size: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Run the Habitat evaluator's batched InternVL preprocessing on CUDA."""

    frames = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError("CUDA preprocessing requires history frames with the same shape")
    pixel_values = torch.from_numpy(np.stack(frames, axis=0)).to(
        device, dtype=torch.float32, non_blocking=True
    ).permute(0, 3, 1, 2).contiguous()
    _, _, height, width = pixel_values.shape
    if height <= width:
        resized_height, resized_width = input_size, max(input_size, int(input_size * width / height))
    else:
        resized_height, resized_width = max(input_size, int(input_size * height / width)), input_size
    top = (resized_height - input_size) // 2
    left = (resized_width - input_size) // 2
    if (height, width) != (resized_height, resized_width):
        pixel_values = F.interpolate(
            pixel_values,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    pixel_values = pixel_values[..., top : top + input_size, left : left + input_size]
    return pixel_values.clamp_(0.0, 255.0).div_(255.0).sub_(mean).div_(std)


def get_cuda_internvl_item(
    dataset: TrackVLADataset,
    index: int,
    device: torch.device,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, Any], Image.Image]:
    """Load one InternVL item while reusing decoded images for rendering."""

    example, base_root = dataset._read_example(index)
    frames_file, current_frame = example.get("frames_file"), example.get("current")
    if not isinstance(frames_file, str) or not isinstance(current_frame, str):
        raise KeyError(f"TrackVLA example {index} requires string frames_file and current fields")
    _, images, _ = dataset._load_history_images(Path(base_root) / Path(frames_file), current_frame)
    token_counts = dataset.history_compressor_config.frame_token_counts(
        len(images), dataset.num_image_tokens
    )
    image_tokens = "\n".join(
        "<img>" + "<IMG_CONTEXT>" * count + "</img>" for count in token_counts
    )
    tokenized = dataset.tokenizer(
        f"{image_tokens}\nUser: {dataset._effective_prompt(example)}\nAssistant:",
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=dataset.max_length,
    )
    return (
        {
            "pixel_values": internvl_cuda_pixel_values(images, input_size, device, mean, std),
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "trajectory": dataset._trajectory_from_example(example, index),
        },
        example,
        images[-1],
    )

def predict_trajectory(
    module: VLAPlModule,
    sample: Dict[str, Any],
    dtype: torch.dtype,
    use_autocast: bool,
) -> np.ndarray:
    batch = prepare_model_batch(sample, module.model_type, module.device)
    autocast_context: contextlib.AbstractContextManager[Any]
    if use_autocast and module.device.type == "cuda":
        autocast_context = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_context = contextlib.nullcontext()
    with torch.inference_mode(), autocast_context:
        outputs = module(batch, extension_stage=ExecutionStage.OFFLINE_EVAL)
    return outputs["pred_traj"].float().cpu().numpy()[0]


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """Return XY EPE/ADE, final displacement error and mean yaw error."""

    xy_error = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1)
    epe = float(xy_error.mean())
    fde = float(xy_error[-1])
    if prediction.shape[1] < 3 or target.shape[1] < 3:
        return epe, fde, float("nan")
    yaw_delta = np.arctan2(
        np.sin(prediction[:, 2] - target[:, 2]),
        np.cos(prediction[:, 2] - target[:, 2]),
    )
    return epe, fde, float(np.degrees(np.abs(yaw_delta).mean()))


def as_polyline_points(trajectory: np.ndarray) -> np.ndarray:
    """Prepend the current ego pose and return an (N, 2) float array."""

    return np.vstack((np.zeros((1, 2), dtype=np.float32), trajectory[:, :2].astype(np.float32)))


def project_homography(points_xy: np.ndarray, homography: np.ndarray) -> Optional[np.ndarray]:
    points_h = np.column_stack((points_xy, np.ones(len(points_xy), dtype=np.float32)))
    projected = (homography @ points_h.T).T
    valid = np.abs(projected[:, 2]) > 1e-6
    if not np.all(valid):
        return None
    return projected[:, :2] / projected[:, 2:3]


def draw_projected_trajectory(
    image_bgr: np.ndarray,
    trajectory: np.ndarray,
    homography: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    points = project_homography(as_polyline_points(trajectory), homography)
    if points is None:
        return
    points_i = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [points_i], isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)
    for point in points_i[:, 0]:
        cv2.circle(image_bgr, tuple(point), 4, color, thickness=-1, lineType=cv2.LINE_AA)


def blend_panel(image_bgr: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (left, top), (left + width, top + height), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.68, image_bgr, 0.32, 0.0, dst=image_bgr)


def draw_ego_inset(
    image_bgr: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    position: str,
    trajectory_range: Tuple[float, float],
) -> None:
    """Draw a fixed-scale ego-coordinate inset when no calibration exists."""

    height, width = image_bgr.shape[:2]
    panel_width = min(max(260, int(width * 0.29)), width - 24)
    panel_height = min(max(210, int(height * 0.28)), height - 24)
    if position == "bottom-center":
        left = (width - panel_width) // 2
        top = height - panel_height - 12
    elif position == "center":
        left = (width - panel_width) // 2
        top = (height - panel_height) // 2
    elif position == "bottom-right":
        left = width - panel_width - 12
        top = height - panel_height - 12
    else:
        raise ValueError(f"Unsupported inset position: {position}")

    max_forward, max_lateral = trajectory_range
    if max_forward <= 0 or max_lateral <= 0:
        raise ValueError("trajectory_range values must be positive")
    origin = np.array([left + panel_width // 2, top + panel_height - 26], dtype=np.float32)
    scale = min((panel_height - 50) / max_forward, (panel_width - 36) / (2.0 * max_lateral))
    blend_panel(image_bgr, left, top, panel_width, panel_height)

    # x is forward/up; y is left, hence negative image x displacement for +y.
    def to_pixels(points: np.ndarray) -> np.ndarray:
        pixels = np.empty_like(points[:, :2], dtype=np.float32)
        pixels[:, 0] = origin[0] - points[:, 1] * scale
        pixels[:, 1] = origin[1] - points[:, 0] * scale
        pixels[:, 0] = np.clip(pixels[:, 0], left + 10, left + panel_width - 10)
        pixels[:, 1] = np.clip(pixels[:, 1], top + 28, origin[1])
        return np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)

    for metre in range(1, int(np.floor(max_forward)) + 1):
        y = int(round(origin[1] - metre * scale))
        cv2.line(image_bgr, (left + 10, y), (left + panel_width - 10, y), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f"{metre}m", (left + 12, max(top + 38, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.line(image_bgr, (int(origin[0]), top + 28), (int(origin[0]), int(origin[1])), (100, 100, 100), 1, cv2.LINE_AA)
    for metre in range(1, int(np.floor(max_lateral)) + 1):
        for sign in (-1, 1):
            x = int(round(origin[0] - sign * metre * scale))
            cv2.line(image_bgr, (x, int(origin[1])), (x, int(origin[1] - 6)), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(image_bgr, f"{sign * metre:+d}", (x - 10, int(origin[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1, cv2.LINE_AA)

    for trajectory, color in ((gt, GT_COLOR), (prediction, PRED_COLOR)):
        pixels = to_pixels(as_polyline_points(trajectory))
        cv2.polylines(image_bgr, [pixels], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
        for point in pixels[1:, 0]:
            cv2.circle(image_bgr, tuple(point), 3, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, tuple(origin.astype(int)), 6, EGO_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(
        image_bgr,
        f"Ego trajectory [m]  1m grid | x:0-{max_forward:.1f}, y:+/-{max_lateral:.1f}",
        (left + 10, top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )



def load_font(preferred_font: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: List[Path] = []
    if preferred_font:
        candidates.append(Path(preferred_font))
    candidates.extend(
        [
            Path(sys.prefix) / "fonts" / "NotoSansCJKsc-Regular.otf",
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
            Path("/Library/Fonts/PingFang.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Width-aware wrapping that also works for Chinese prompts without spaces."""

    lines: List[str] = []
    current = ""
    for character in text.replace("\n", " "):
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def draw_metadata(
    image_bgr: np.ndarray,
    frame_id: str,
    sample_index: int,
    prompt: str,
    epe: float,
    fde: float,
    yaw_mae: float,
    font_path: Optional[str],
    direct_projection: bool,
) -> np.ndarray:
    """Add the upper-left translucent metadata panel using a Unicode-capable font."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(font_path, max(16, image_bgr.shape[1] // 90))
    small_font = load_font(font_path, max(14, image_bgr.shape[1] // 108))
    margin = 12
    max_text_width = min(int(image_bgr.shape[1] * 0.72), 1050)
    prompt_lines = wrap_text(draw, f"prompt: {prompt}", small_font, max_text_width)
    lines = [
        f"frame_id={frame_id}   sample={sample_index}",
        f"EPE/ADE={epe:.3f} m   FDE={fde:.3f} m   yaw MAE={yaw_mae:.1f} deg",
        *prompt_lines,
        "Yellow=GT   Cyan=Pred   Green=current ego"
        + ("   (projected on image)" if direct_projection else "   (ego-frame inset)"),
    ]
    line_heights = [
        int((font if index < 2 else small_font).getbbox(line)[3] - (font if index < 2 else small_font).getbbox(line)[1]) + 7
        for index, line in enumerate(lines)
    ]
    panel_height = sum(line_heights) + 2 * margin
    panel_width = min(image_bgr.shape[1] - 2 * margin, max_text_width + 2 * margin)
    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=8,
        fill=(0, 0, 0, 155),
    )
    y = margin + 7
    for index, (line, line_height) in enumerate(zip(lines, line_heights)):
        line_font = font if index < 2 else small_font
        draw.text((margin + 9, y), line, fill=(255, 255, 255, 255), font=line_font)
        y += line_height
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def frame_id_from_name(current_name: str) -> str:
    match = re.search(r"(\d+)(?!.*\d)", Path(current_name).stem)
    return str(int(match.group(1))) if match else current_name


def upscale_output_frame(image_bgr: np.ndarray, minimum_height: int = 1080) -> np.ndarray:
    """Upscale rendering only, preserving source aspect ratio."""

    height, width = image_bgr.shape[:2]
    if height >= minimum_height:
        return image_bgr
    scale = minimum_height / height
    return cv2.resize(
        image_bgr,
        (int(round(width * scale)), minimum_height),
        interpolation=cv2.INTER_CUBIC,
    )

def iter_indices(length: int, start: int, stride: int, max_frames: Optional[int]) -> Iterable[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start < 0 or start >= length:
        raise ValueError(f"--start-index must be in [0, {length - 1}], got {start}")
    rendered = 0
    for index in range(start, length, stride):
        if max_frames is not None and rendered >= max_frames:
            break
        yield index
        rendered += 1


def render_video(args: argparse.Namespace) -> None:
    frames_path = Path(args.frames).expanduser().resolve()
    if not frames_path.is_file():
        raise FileNotFoundError(f"--frames does not exist: {frames_path}")
    samples_path = resolve_samples_path(frames_path, args.samples)
    device = select_device(args.device)
    print(f"[vis_trackvla] frames: {frames_path}")
    print(f"[vis_trackvla] samples: {samples_path}")
    print(f"[vis_trackvla] device: {device}")
    if args.config is None:
        print(f"[vis_trackvla] config: {checkpoint_config_path(args.checkpoint)}")
    else:
        print(f"[vis_trackvla] config override: {Path(args.config).expanduser()}")

    module, config, dtype = load_model(args.checkpoint, args.config, device)
    dataset = build_dataset(module, config, samples_path)
    use_cuda_transform = (
        device.type == "cuda"
        and module.model_type in {"internvl", "internvl3"}
        and not args.cpu_transform
        and not dataset.load_history_masks
    )
    input_size = int(config["data"]["input_size"])
    internvl_mean = internvl_std = None
    if use_cuda_transform:
        internvl_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        internvl_std = torch.tensor(
            [0.229, 0.224, 0.225], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        print("[vis_trackvla] using Habitat-style CUDA image preprocessing")
    indices = list(iter_indices(len(dataset), args.start_index, args.stride, args.max_frames))
    if not indices:
        raise ValueError("No samples selected for rendering")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    homography = np.asarray(args.homography, dtype=np.float32).reshape(3, 3) if args.homography else None
    writer: Optional[cv2.VideoWriter] = None
    all_epes: List[float] = []
    all_fdes: List[float] = []

    try:
        for rendered_count, index in enumerate(indices, start=1):
            if use_cuda_transform:
                assert internvl_mean is not None and internvl_std is not None
                sample, example, raw_image = get_cuda_internvl_item(
                    dataset, index, device, input_size, internvl_mean, internvl_std
                )
            else:
                # Keep the Dataset path for CPU, Qwen2-VL, and mask-conditioned models.
                sample = dataset.get_item(index, augment_history_masks=False)
                example, base_root = dataset._read_example(index)
                archive_path = Path(base_root) / Path(example["frames_file"])
                _, images, _ = dataset._load_history_images(archive_path, example["current"])
                raw_image = images[-1]
            target = sample["trajectory"].numpy()
            prediction = predict_trajectory(module, sample, dtype, not args.no_autocast)
            epe, fde, yaw_mae = trajectory_metrics(prediction, target)
            all_epes.append(epe)
            all_fdes.append(fde)

            image_bgr = cv2.cvtColor(np.asarray(raw_image), cv2.COLOR_RGB2BGR)
            if not args.no_upscale:
                image_bgr = upscale_output_frame(image_bgr)
            if homography is not None:
                draw_projected_trajectory(image_bgr, target, homography, GT_COLOR)
                draw_projected_trajectory(image_bgr, prediction, homography, PRED_COLOR)
                ego_pixel = project_homography(np.zeros((1, 2), dtype=np.float32), homography)
                if ego_pixel is not None:
                    cv2.circle(image_bgr, tuple(np.rint(ego_pixel[0]).astype(int)), 7, EGO_COLOR, -1, cv2.LINE_AA)
            else:
                draw_ego_inset(
                    image_bgr, target, prediction, args.inset_position, tuple(args.trajectory_range)
                )

            prompt = dataset._effective_prompt(example)
            image_bgr = draw_metadata(
                image_bgr,
                frame_id=frame_id_from_name(str(example["current"])),
                sample_index=index,
                prompt=prompt,
                epe=epe,
                fde=fde,
                yaw_mae=yaw_mae,
                font_path=args.font,
                direct_projection=homography is not None,
            )
            if writer is None:
                height, width = image_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open video writer for {output_path}. "
                        "Use an .mp4 output path and ensure OpenCV has an MP4 encoder."
                    )
            writer.write(image_bgr)
            print(
                f"[vis_trackvla] {rendered_count}/{len(indices)} index={index} "
                f"EPE={epe:.3f}m FDE={fde:.3f}m"
            )
    finally:
        if writer is not None:
            writer.release()

    print(
        f"[vis_trackvla] wrote {output_path} | frames={len(all_epes)} "
        f"mean EPE/ADE={np.mean(all_epes):.3f}m mean FDE={np.mean(all_fdes):.3f}m"
    )


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")
    if any(value <= 0 for value in args.trajectory_range):
        raise ValueError("--trajectory-range values must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    render_video(args)


if __name__ == "__main__":
    main()
