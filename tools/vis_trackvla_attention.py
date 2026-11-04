#!/usr/bin/env python3
"""Offline attention and layer-ablation diagnostics for InternVL3 TrackVLA.

The tool intentionally reconstructs the model from the checkpoint's saved
configuration, but loads its decoder with eager attention so per-layer
attention probabilities are available.  It never changes a checkpoint or the
normal training/model API.

Example:
    python tools/vis_trackvla_attention.py \
        --frames data/trackvla/at_train_60/frames/.../frames.jpgs \
        --checkpoint outputs/run/checkpoints/last.ckpt \
        --output-dir outputs/attention_run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla.config import ExecutionStage, ModelExtensionContext, ModelExtensionsConfig
from vla.data.trackvla import TrackVLADataset
from vla.lightning import VLAPlModule
from vla.utils.attention_diagnostics import (
    AttentionTokenLayout,
    LLMFrameAttentionMaps,
    LLMAttentionCollector,
    QFormerAttentionCollector,
    compose_llm_qformer_attention,
    layer_cutoff_indices,
    resolve_decoder_layers,
    temporary_llm_attention_capture,
    temporary_qformer_attention_capture,
    temporary_visual_token_ablation,
)
from vla.utils.attention_plotting import (
    render_attention_overview,
    render_attention_summary,
    render_intervention_summary,
    render_qformer_attention,
    render_spatial_attention_overlays,
)
from vla.utils.checkpoint_config import (
    checkpoint_config_path,
    load_checkpoint_config,
    load_checkpoint_hparams,
)
from vla.utils.common import ResizeLongestSideAndPad, build_transform
from vla.utils.trackvla_attention_data import (
    DEFAULT_MAX_SAMPLES,
    ResolvedTrackVLAEpisode,
    resolve_trackvla_episode,
    select_episode_indices,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize InternVL3 TrackVLA prompt/action-query attention and "
            "run layer-depth visual-token ablations for one frames.jpgs archive."
        )
    )
    parser.add_argument("--frames", required=True, help="TrackVLA frames.jpgs archive.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning .ckpt file (or a checkpoint supported by VLAPlModule).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic reports.")
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional dataset.json or JSONL override. It must reference --frames.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training YAML override; defaults to the config associated with --checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model.model_path from the saved configuration.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated episode-local manifest positions; mutually exclusive with range selection.",
    )
    parser.add_argument("--start-index", type=int, default=None, help="Episode-local start index.")
    parser.add_argument("--stride", type=int, default=None, help="Episode-local sampling stride.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=f"Maximum selected records (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, cuda:0, or cpu (default: auto).",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable CUDA bf16/fp16 autocast during diagnostic forwards.",
    )
    parser.add_argument(
        "--ablation-cutoffs",
        default=None,
        help=(
            "Optional comma-separated decoder-layer cutoffs. Defaults to "
            "0,25%%,50%%,75%%,100%% of the decoder depth."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for batch summaries (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default: 0).")
    return parser.parse_args(argv)


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested with --device {value}, but CUDA is unavailable")
    return device


def _dtype_from_config(value: object) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    normalized = str(value or "bf16").lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported model dtype {value!r}")
    return mapping[normalized]


def load_attention_model(
    checkpoint: str,
    config_path: Optional[str],
    model_path_override: Optional[str],
    device: torch.device,
) -> Tuple[VLAPlModule, Dict[str, Any], torch.dtype]:
    """Restore one InternVL3 checkpoint with eager decoder attention."""

    config = load_checkpoint_config(checkpoint, config_path)
    if config.get("data", {}).get("dataset_type") != "trackvla":
        raise ValueError(
            "This tool accepts TrackVLA archives, so config.data.dataset_type must be 'trackvla'."
        )
    model_config = config.get("model")
    data_config = config.get("data")
    if not isinstance(model_config, dict) or not isinstance(data_config, dict):
        raise ValueError("Checkpoint configuration must contain mapping-valued model and data sections")

    checkpoint_hparams = load_checkpoint_hparams(checkpoint)
    model_type = model_config.get("model_type") or checkpoint_hparams.get("model_type")
    if model_type != "internvl3":
        raise ValueError(
            "vis_trackvla_attention.py currently supports only model_type='internvl3', "
            f"got {model_type!r}."
        )
    extensions = ModelExtensionsConfig.resolve_checkpoint(
        model_config=model_config,
        checkpoint_hparams=checkpoint_hparams,
        context=ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type="trackvla",
            model_type="internvl3",
            dense_tokens_per_frame=int(data_config.get("num_image_tokens", 256)),
        ),
    )
    extensions.validate_for(
        ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type="trackvla",
            model_type="internvl3",
            dense_tokens_per_frame=int(data_config.get("num_image_tokens", 256)),
        )
    )
    extensions.apply_to_model_mapping(model_config)
    dtype = _dtype_from_config(model_config.get("dtype", "bf16"))
    model_path = model_path_override or model_config.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("No usable model path; pass --model-path or configure model.model_path")

    module = VLAPlModule.load_from_checkpoint(
        checkpoint,
        model_path=model_path,
        model_type="internvl3",
        num_queries=int(model_config["num_queries"]),
        n_future=int(data_config["n_future"]),
        freeze_vision=bool(model_config["freeze_vision"]),
        freeze_llm=bool(model_config["freeze_llm"]),
        freeze_mlp1=bool(model_config.get("freeze_mlp1", True)),
        freeze_action_head=bool(model_config.get("freeze_action_head", False)),
        action_head_dims=tuple(model_config["action_head_dims"]),
        action_head_type=str(model_config.get("action_head_type", "legacy")),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        max_steps=int(config["training"]["max_steps"]),
        yaw_weight=float(config["training"].get("yaw_weight", 1.0)),
        x_weight=float(config["training"].get("x_weight", 1.0)),
        y_weight=float(config["training"].get("y_weight", 1.0)),
        dtype=dtype,
        # SDPA and FlashAttention do not expose attention matrices.  This is
        # intentionally isolated to the diagnostic model instance.
        attn_implementation="eager",
        sam2_config=extensions.sam2,
        history_compressor_config=extensions.history_compressor,
        auxiliary_tasks_config=model_config.get("auxiliary_tasks_config"),
    )
    module.to(device)
    module.eval()
    module.freeze()
    return module, config, dtype


def build_dataset(
    module: VLAPlModule,
    config: Mapping[str, Any],
    episode: ResolvedTrackVLAEpisode,
) -> TrackVLADataset:
    data_config = config["data"]
    extensions = module.model_extensions
    return TrackVLADataset(
        source_paths=str(episode.manifest_path),
        tokenizer=module.model.tokenizer,
        transform=build_transform(int(data_config["input_size"])),
        num_history_frames=int(data_config["num_history_frames"]),
        history_frame_interval=int(data_config.get("history_frame_interval", 0)),
        n_future=int(data_config["n_future"]),
        trajectory_dim=int(data_config.get("trajectory_dim", 3)),
        prompt=str(data_config.get("prompt", "")),
        num_image_tokens=int(data_config.get("num_image_tokens", 256)),
        max_length=int(data_config.get("max_length", 1200)),
        model_type="internvl3",
        index_cache_dir=None,
        max_open_archives=1,
        # Offline Q-Former consumes clean historical masks; when SAM2 is on,
        # the Lightning offline contract also requires the current GT mask.
        load_masks=bool(extensions.sam2.enabled),
        auxiliary_tasks_config=module.auxiliary_tasks_config,
        mask_input_size=int(data_config["input_size"]),
        history_compressor_config=extensions.history_compressor,
        history_mask_augmentation=False,
        # ``--samples`` may be an explicit manifest outside the subset.  Keep
        # its record order but resolve frame/mask paths against --frames' root.
        base_root_override=episode.subset_root,
    )


def prepare_model_batch(sample: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {
        "pixel_values": sample["pixel_values"].to(device),
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }
    if "mask" in sample:
        batch["masks"] = sample["mask"].unsqueeze(0).to(device)
    if "history_masks" in sample:
        batch["history_masks"] = sample["history_masks"].unsqueeze(0).to(device)
        batch["history_mask_valid"] = sample["history_mask_valid"].unsqueeze(0).to(device)
    for key in ("history_boxes", "history_box_valid", "grounding_bbox_target", "grounding_visibility_target", "vqa_labels"):
        if key in sample:
            batch[key] = sample[key].unsqueeze(0).to(device)
    return batch


def forward_module(
    module: VLAPlModule,
    batch: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    no_autocast: bool,
) -> Mapping[str, torch.Tensor]:
    if module.device.type == "cuda" and not no_autocast and dtype in (torch.bfloat16, torch.float16):
        autocast: contextlib.AbstractContextManager[Any] = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast = contextlib.nullcontext()
    with torch.inference_mode(), autocast:
        return module(dict(batch), extension_stage=ExecutionStage.OFFLINE_EVAL)


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    xy_error = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1)
    result = {"epe": float(xy_error.mean()), "fde": float(xy_error[-1])}
    if prediction.shape[-1] >= 3 and target.shape[-1] >= 3:
        yaw = np.arctan2(
            np.sin(prediction[:, 2] - target[:, 2]),
            np.cos(prediction[:, 2] - target[:, 2]),
        )
        result["yaw_mae_deg"] = float(np.degrees(np.abs(yaw)).mean())
    else:
        result["yaw_mae_deg"] = float("nan")
    return result


def prediction_drift(prediction: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    xy_drift = np.linalg.norm(prediction[:, :2] - baseline[:, :2], axis=1)
    result = {"waypoint_drift": float(xy_drift.mean()), "final_drift": float(xy_drift[-1])}
    if prediction.shape[-1] >= 3 and baseline.shape[-1] >= 3:
        yaw = np.arctan2(
            np.sin(prediction[:, 2] - baseline[:, 2]),
            np.cos(prediction[:, 2] - baseline[:, 2]),
        )
        result["yaw_drift_deg"] = float(np.degrees(np.abs(yaw)).mean())
    else:
        result["yaw_drift_deg"] = float("nan")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_cutoffs(value: Optional[str], decoder_depth: int) -> List[int]:
    if decoder_depth <= 0:
        raise ValueError("Decoder must expose at least one layer for layer-depth ablation")
    if value is None:
        return layer_cutoff_indices(decoder_depth)
    values: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--ablation-cutoffs must be a comma-separated list of layer indices")
        try:
            cutoff = int(item)
        except ValueError as exc:
            raise ValueError("--ablation-cutoffs must contain integer layer indices") from exc
        if cutoff < 0 or cutoff > decoder_depth:
            raise ValueError(f"Ablation cutoff {cutoff} must be in [0, {decoder_depth}]")
        values.append(cutoff)
    if not values:
        raise ValueError("At least one ablation cutoff is required")
    return sorted(set(values))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a 95% bootstrap CI over the first/sample axis."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("bootstrap requires at least one sample")
    if values.shape[0] == 1 or resamples <= 0:
        mean = values.mean(axis=0)
        return mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    sample_count = values.shape[0]
    indices = rng.integers(0, sample_count, size=(resamples, sample_count))
    means = values[indices].mean(axis=1)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def _sample_images(dataset: TrackVLADataset, index: int) -> Tuple[Mapping[str, Any], List[Any], List[str]]:
    example, base_root = dataset._read_example(index)
    frames_file = example.get("frames_file")
    current = example.get("current")
    if not isinstance(frames_file, str) or not isinstance(current, str):
        raise KeyError(f"TrackVLA sample {index} is missing frames_file/current")
    names, images, _ = dataset._load_history_images(Path(base_root) / Path(frames_file), current)
    return example, images, names


def _model_view_images(images: Sequence[Any], input_size: int) -> List[Any]:
    """Apply the geometric (not tensor/normalization) model preprocessing.

    Attention maps live on the resized and padded square seen by InternVL3.
    Rendering them over the source camera RGB would geometrically misregister
    non-square frames, so reports intentionally show this exact model view.
    """
    display_transform = ResizeLongestSideAndPad(input_size)
    return [display_transform(image.convert("RGB")) for image in images]


def _layout_for_sample(module: VLAPlModule, dataset: TrackVLADataset, sample: Mapping[str, Any]) -> AttentionTokenLayout:
    input_ids = sample["input_ids"].detach().cpu()
    attention_mask = sample["attention_mask"].detach().cpu()
    valid_length = int(attention_mask.to(dtype=torch.long).sum().item())
    if valid_length != int(input_ids.numel()):
        # The CLI deliberately processes one unpadded sample at a time.  This
        # preserves a one-to-one token-index relationship with the decoder
        # attention matrices and gives a useful error if a caller changes that.
        raise ValueError("Attention diagnostics expects an unpadded single sample")
    frame_counts = dataset.history_compressor_config.frame_token_counts(
        dataset.num_history_frames,
        dataset.num_image_tokens,
    )
    visual_positions = torch.nonzero(
        input_ids == int(module.model.img_context_token_id), as_tuple=False
    ).flatten().tolist()
    if len(visual_positions) != sum(frame_counts):
        raise ValueError(
            "Image placeholder count does not match the compressor-aware frame layout: "
            f"got {len(visual_positions)}, expected {sum(frame_counts)}"
        )

    def contiguous_spans(positions: Sequence[int]) -> tuple[Any, ...]:
        from vla.utils.attention_diagnostics import TokenSpan

        if not positions:
            raise ValueError("Every image frame needs at least one visual token")
        result = []
        start = previous = int(positions[0])
        for position in positions[1:]:
            position = int(position)
            if position != previous + 1:
                result.append(TokenSpan(start, previous + 1))
                start = position
            previous = position
        result.append(TokenSpan(start, previous + 1))
        return tuple(result)

    cursor = 0
    frame_spans = []
    for count in frame_counts:
        frame_spans.append(contiguous_spans(visual_positions[cursor : cursor + count]))
        cursor += count
    prompt_length = int(input_ids.numel())
    mask_query_count = int(getattr(module.model, "mask_query_token", None) is not None)
    action_start = prompt_length + mask_query_count
    from vla.utils.attention_diagnostics import TokenSpan

    return AttentionTokenLayout.from_visual_spans(
        sequence_length=prompt_length + mask_query_count + int(module.model.num_queries),
        history_frame_spans=frame_spans[:-1],
        current_frame_spans=frame_spans[-1],
        action_query_spans=TokenSpan(action_start, action_start + int(module.model.num_queries)),
        mask_query_spans=(TokenSpan(prompt_length, action_start) if mask_query_count else None),
    )


def _selected_layer_labels(layer_count: int) -> Tuple[List[int], List[str]]:
    selected = sorted(set((0, max(0, (layer_count - 1) // 2), layer_count - 1)))
    label_names = {selected[0]: "early", selected[-1]: "late"}
    if len(selected) == 3:
        label_names[selected[1]] = "middle"
    return selected, [f"{label_names[index]} L{index}" for index in selected]


def _aggregation_mapping(aggregation: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Convert the core `[L,H,Q,S]` aggregation into plotting-friendly maps."""

    mass = aggregation.mass.detach().cpu().numpy()
    density = aggregation.density.detach().cpu().numpy()
    group_names = list(aggregation.query_group_names)
    return (
        {name: mass[:, :, index, :] for index, name in enumerate(group_names)},
        {name: density[:, :, index, :] for index, name in enumerate(group_names)},
        list(aggregation.source_group_names),
    )


def _visual_source_indices(source_labels: Sequence[str]) -> List[int]:
    return [
        index
        for index, label in enumerate(source_labels)
        if label.startswith("history_") or label == "current"
    ]


def _run_interventions(
    module: VLAPlModule,
    batch: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    no_autocast: bool,
    layout: AttentionTokenLayout,
    baseline: np.ndarray,
    target: np.ndarray,
    cutoffs: Sequence[int],
) -> Dict[str, List[Dict[str, float]]]:
    decoder_layers = resolve_decoder_layers(module.model.internvl.language_model)
    scopes: Dict[str, tuple[Any, ...]] = {}
    history_spans = layout.visual_spans("history")
    if history_spans:
        scopes["history"] = history_spans
    scopes.update(
        {
            "current": layout.visual_spans("current"),
            "all_visual": layout.visual_spans("all"),
        }
    )
    results: Dict[str, List[Dict[str, float]]] = {scope: [] for scope in scopes}
    for scope, visual_spans in scopes.items():
        for cutoff in cutoffs:
            with temporary_visual_token_ablation(
                decoder_layers, visual_spans, cutoff_layer=cutoff
            ):
                outputs = forward_module(module, batch, dtype, no_autocast)
            prediction = outputs["pred_traj"].float().cpu().numpy()[0]
            drift = prediction_drift(prediction, baseline)
            metrics = trajectory_metrics(prediction, target)
            results[scope].append({"cutoff": int(cutoff), **drift, **metrics})
    return results


def _write_sample_reports(
    *,
    output_dir: Path,
    sample_position: int,
    source_index: int,
    frame_names: Sequence[str],
    raw_images: Sequence[Any],
    layout: AttentionTokenLayout,
    aggregation: Any,
    frame_maps: LLMFrameAttentionMaps,
    qformer_attention: Optional[Any],
    qformer_status: str,
    target: np.ndarray,
    prediction: np.ndarray,
    intervention: Mapping[str, Any],
) -> Dict[str, Any]:
    sample_dir = output_dir / "samples" / f"{sample_position:04d}_manifest_{source_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    mass, density, source_labels = _aggregation_mapping(aggregation)
    render_attention_overview(
        sample_dir / "attention_overview.png",
        mass,
        source_labels=source_labels,
        attention_density=density,
        title=f"sample {sample_position} | prompt/action attention",
    )

    layer_count = int(aggregation.num_layers)
    selected_layers, layer_labels = _selected_layer_labels(layer_count)
    spatial_metadata: Dict[str, str] = {}
    frame_groups = layout.frame_groups
    if len(raw_images) != len(frame_groups):
        raise ValueError(
            "Loaded image count does not match the compressor-aware visual token layout: "
            f"got {len(raw_images)} images and {len(frame_groups)} frame groups"
        )
    query_group_indices = {
        name: index for index, name in enumerate(frame_maps.query_group_names)
    }
    for group in layout.query_groups:
        try:
            query_group_index = query_group_indices[group.name]
        except KeyError as exc:
            raise RuntimeError(
                f"Captured LLM maps omit expected query group {group.name!r}"
            ) from exc
        for frame_index, image in enumerate(raw_images):
            frame_group = frame_groups[frame_index]
            if qformer_attention is not None and frame_group.name.startswith("history_"):
                spatial_metadata[f"{group.name}_{frame_group.name}"] = (
                    "skipped: learned Q-Former query tokens have no fixed spatial grid; "
                    "use the composed full-patch overlay"
                )
                continue
            if frame_maps.grid_sizes[frame_group.name] is None:
                spatial_metadata[f"{group.name}_{frame_group.name}"] = (
                    "skipped: non-square visual-token count; token-level data is in attentions.npz"
                )
                continue
            direct = frame_maps.spatial_map(frame_group.name)[:, query_group_index]
            direct = direct.detach().cpu().numpy()
            maps = direct[selected_layers]
            label = frame_group.name
            destination = sample_dir / f"{group.name}_{label}_spatial.png"
            render_spatial_attention_overlays(
                destination,
                image,
                maps,
                labels=layer_labels,
                title=f"{group.name} -> {label} | direct LLM attention (model-input crop)",
            )
            spatial_metadata[f"{group.name}_{label}"] = destination.name

    qformer_files: List[str] = []
    composed_attention: Dict[str, np.ndarray] = {}
    if qformer_attention is not None:
        for frame_index, image in enumerate(raw_images[:-1]):
            for q_layer, query_maps in enumerate(qformer_attention[frame_index]):
                destination = sample_dir / f"qformer_history_{frame_index}_layer_{q_layer}.png"
                render_qformer_attention(
                    destination,
                    image,
                    query_maps,
                    title=(
                        f"Q-Former history {frame_index} | compressor layer {q_layer} "
                        "(model-input crop)"
                    ),
                )
                qformer_files.append(destination.name)
            # Compose each LLM layer's attention to the compressed history
            # queries with the *last* Q-Former full-feature cross-attention.
            # This is a visual attention composition, not a causal attribution.
            qformer_last = qformer_attention[frame_index, -1].mean(dim=0)
            history_group = layout.history_frames[frame_index]
            for group in layout.query_groups:
                query_group_index = query_group_indices[group.name]
                weights = frame_maps.token_map(history_group.name)[:, query_group_index]
                composed = compose_llm_qformer_attention(weights, qformer_last).numpy()
                composed_attention[f"{group.name}_to_{history_group.name}"] = composed
                destination = sample_dir / f"{group.name}_{history_group.name}_composed.png"
                render_spatial_attention_overlays(
                    destination,
                    image,
                    composed[selected_layers],
                    labels=layer_labels,
                    title=(
                        f"{group.name} -> {history_group.name} | "
                        "LLM/Q-Former attention composition (model-input crop)"
                    ),
                )
                spatial_metadata[f"{group.name}_{history_group.name}_composed"] = destination.name

    arrays: Dict[str, np.ndarray] = {}
    for group, values in mass.items():
        arrays[f"{group}_mass"] = np.asarray(values)
        arrays[f"{group}_density"] = np.asarray(density[group])
    for frame_name in frame_maps.frame_names:
        arrays[f"llm_to_{frame_name}_token_attention"] = (
            frame_maps.token_map(frame_name).detach().cpu().numpy()
        )
    arrays.update(
        {
            f"composed_{name}": values
            for name, values in composed_attention.items()
        }
    )
    if qformer_attention is not None:
        arrays["qformer_full_attention"] = qformer_attention.detach().cpu().numpy()
    np.savez_compressed(sample_dir / "attentions.npz", **arrays)
    metrics = {
        "sample_position": sample_position,
        "manifest_record_index": source_index,
        "frame_names": list(frame_names),
        "token_layout": layout.to_dict(),
        "frame_attention_layout": frame_maps.to_dict(),
        "target": target,
        "prediction": prediction,
        "baseline_metrics": trajectory_metrics(prediction, target),
        "intervention": intervention,
        "intervention_scopes": list(intervention),
        "spatial_files": spatial_metadata,
        "qformer_files": qformer_files,
        "qformer_enabled": qformer_attention is not None,
        "qformer_status": qformer_status,
        "qformer_attention_shape": (
            None if qformer_attention is None else list(qformer_attention.shape)
        ),
        "qformer_composition_note": (
            "LLM-to-compressed-query and Q-Former query-to-full-patch maps are "
            "an attention composition, not a strict causal attribution."
            if qformer_attention is not None
            else None
        ),
        "source_labels": source_labels,
    }
    write_json(sample_dir / "metrics.json", metrics)
    return metrics


def _aggregate_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("No sample reports were generated")
    source_labels = records[0]["source_labels"]
    visual_indices = _visual_source_indices(source_labels)
    summary: Dict[str, Any] = {"samples": len(records), "source_labels": source_labels}
    series: Dict[str, np.ndarray] = {}
    history_indices = [
        index for index, label in enumerate(source_labels) if label.startswith("history_")
    ]
    current_indices = [
        index for index, label in enumerate(source_labels) if label == "current"
    ]
    query_groups = ("instruction_prompt", "action_queries")
    for group in query_groups:
        total_values = []
        history_values = []
        current_values = []
        for record in records:
            attention_path = (
                output_dir
                / "samples"
                / f"{record['sample_position']:04d}_manifest_{record['manifest_record_index']:06d}"
                / "attentions.npz"
            )
            with np.load(attention_path) as data:
                masses = data[f"{group}_mass"]
                total_values.append(masses[..., visual_indices].sum(axis=-1).mean(axis=-1))
                if history_indices:
                    history_values.append(
                        masses[..., history_indices].sum(axis=-1).mean(axis=-1)
                    )
                current_values.append(
                    masses[..., current_indices].sum(axis=-1).mean(axis=-1)
                )
        series[f"{group}: all visual"] = np.stack(total_values, axis=0)
        if history_values:
            series[f"{group}: history"] = np.stack(history_values, axis=0)
        series[f"{group}: current"] = np.stack(current_values, axis=0)
    ci = {
        name: bootstrap_mean_ci(value, seed=seed, resamples=bootstrap_samples)
        for name, value in series.items()
    }
    summary["attention"] = {
        "series": series,
        "ci95": {
            name: {"low": interval[0], "high": interval[1]}
            for name, interval in ci.items()
        },
    }
    render_attention_summary(
        output_dir / "summary_attention.png",
        series,
        title="Visual attention by LLM depth",
        ylabel="attention mass to visual tokens",
        ci=ci,
    )
    third = max(1, next(iter(series.values())).shape[1] // 3)
    summary["shallow_vs_deep"] = {
        name: {
            "per_sample_difference": (values[:, :third].mean(axis=1) - values[:, -third:].mean(axis=1)),
            "mean_difference": float((values[:, :third].mean(axis=1) - values[:, -third:].mean(axis=1)).mean()),
            "ci95": bootstrap_mean_ci(
                values[:, :third].mean(axis=1) - values[:, -third:].mean(axis=1),
                seed=seed,
                resamples=bootstrap_samples,
            ),
        }
        for name, values in series.items()
    }

    scopes = tuple(records[0]["intervention"])
    if not scopes:
        raise ValueError("No visual-token ablation scopes were generated")
    cutoffs = [item["cutoff"] for item in records[0]["intervention"][scopes[0]]]
    metric_names = ("waypoint_drift", "final_drift", "yaw_drift_deg", "epe", "fde", "yaw_mae_deg")
    intervention_summary: Dict[str, Any] = {"cutoffs": cutoffs, "scopes": list(scopes)}
    chart_metrics: Dict[str, np.ndarray] = {}
    for metric in metric_names:
        values = np.asarray(
            [
                [[entry[metric] for entry in record["intervention"][scope]] for scope in scopes]
                for record in records
            ],
            dtype=np.float64,
        )
        intervention_summary[metric] = values
        chart_metrics[metric] = values
    render_intervention_summary(
        output_dir / "summary_intervention.png",
        chart_metrics,
        cutoffs=cutoffs,
        scopes=scopes,
        title="Visual-token ablation sensitivity",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary["intervention"] = intervention_summary

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_position",
                "manifest_record_index",
                "baseline_epe",
                "baseline_fde",
                "baseline_yaw_mae_deg",
            ),
        )
        writer.writeheader()
        for record in records:
            baseline = record["baseline_metrics"]
            writer.writerow(
                {
                    "sample_position": record["sample_position"],
                    "manifest_record_index": record["manifest_record_index"],
                    "baseline_epe": baseline["epe"],
                    "baseline_fde": baseline["fde"],
                    "baseline_yaw_mae_deg": baseline["yaw_mae_deg"],
                }
            )
    intervention_csv_path = output_dir / "intervention.csv"
    with intervention_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_position",
                "manifest_record_index",
                "scope",
                "cutoff",
                *metric_names,
            ),
        )
        writer.writeheader()
        for record in records:
            for scope in scopes:
                for entry in record["intervention"][scope]:
                    writer.writerow(
                        {
                            "sample_position": record["sample_position"],
                            "manifest_record_index": record["manifest_record_index"],
                            "scope": scope,
                            **entry,
                        }
                    )
    return summary


def run(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint does not exist: {checkpoint}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"--output-dir already contains files: {output_dir}. Choose a new directory to avoid mixing runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    episode = resolve_trackvla_episode(args.frames, args.samples)
    selected_positions = select_episode_indices(
        len(episode.records),
        indices=args.indices,
        start_index=args.start_index,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    if not selected_positions:
        raise ValueError("Sample selection is empty")

    print(f"[attention-vis] checkpoint: {checkpoint}")
    print(f"[attention-vis] frames: {episode.frames_path}")
    print(f"[attention-vis] manifest: {episode.manifest_path} ({episode.source})")
    print(f"[attention-vis] samples: {len(selected_positions)} / {len(episode.records)}")
    module, config, dtype = load_attention_model(
        str(checkpoint), args.config, args.model_path, device
    )
    dataset = build_dataset(module, config, episode)
    decoder_layers = resolve_decoder_layers(module.model.internvl.language_model)
    cutoffs = parse_cutoffs(args.ablation_cutoffs, len(decoder_layers))
    qformer_enabled = bool(module.model.history_compressor_config.requires_history_masks or module.model.history_compressor_config.requires_history_boxes)
    compressor_config = module.model.history_compressor_config
    if not compressor_config.enabled:
        qformer_status = "skipped: history compressor is disabled"
    elif not (compressor_config.requires_history_masks or compressor_config.requires_history_boxes):
        qformer_status = (
            f"skipped: history compressor mode is {compressor_config.mode!r}, not a QFormer mode"
        )
    elif dataset.num_history_frames <= 1:
        qformer_status = "skipped: configuration has no historical frame to compress"
    else:
        qformer_status = f"captured: {compressor_config.mode} full-feature cross-attention"

    reports: List[Mapping[str, Any]] = []
    try:
        for rendered_index, episode_position in enumerate(selected_positions):
            manifest_index = episode.record_indices[episode_position]
            sample = dataset.get_item(manifest_index, augment_history_masks=False)
            example, raw_images, frame_names = _sample_images(dataset, manifest_index)
            display_images = _model_view_images(
                raw_images, int(config["data"]["input_size"])
            )
            batch = prepare_model_batch(sample, device)
            layout = _layout_for_sample(module, dataset, sample)
            appended_count = int(module.model.num_queries) + int(
                getattr(module.model, "mask_query_token", None) is not None
            )
            full_attention_mask = torch.cat(
                (
                    batch["attention_mask"],
                    torch.ones(
                        (1, appended_count),
                        dtype=batch["attention_mask"].dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            llm_collector = LLMAttentionCollector(
                layout,
                attention_mask=full_attention_mask,
                capture_frame_maps=True,
            )
            qformer_collector = QFormerAttentionCollector()
            qformer_context: contextlib.AbstractContextManager[Any]
            qformer_context = (
                temporary_qformer_attention_capture(module.model.history_compressor, qformer_collector)
                if qformer_enabled and len(raw_images) > 1
                else contextlib.nullcontext()
            )
            with temporary_llm_attention_capture(module.model.internvl.language_model, llm_collector):
                with qformer_context:
                    outputs = forward_module(module, batch, dtype, args.no_autocast)
            prediction = outputs["pred_traj"].float().cpu().numpy()[0]
            target = sample["trajectory"].float().cpu().numpy()
            aggregation = llm_collector.latest
            frame_maps = llm_collector.latest_frame_maps
            qformer_attention = None
            if qformer_enabled and len(raw_images) > 1:
                qformer_attention = qformer_collector.patch_maps(
                    num_frames=len(raw_images) - 1
                )[0]
            intervention = _run_interventions(
                module,
                batch,
                dtype,
                args.no_autocast,
                layout,
                prediction,
                target,
                cutoffs,
            )
            report = _write_sample_reports(
                output_dir=output_dir,
                sample_position=episode_position,
                source_index=manifest_index,
                frame_names=frame_names,
                raw_images=display_images,
                layout=layout,
                aggregation=aggregation,
                frame_maps=frame_maps,
                qformer_attention=qformer_attention,
                qformer_status=qformer_status,
                target=target,
                prediction=prediction,
                intervention=intervention,
            )
            reports.append(report)
            print(
                f"[attention-vis] {rendered_index + 1}/{len(selected_positions)} "
                f"episode_index={episode_position} manifest_index={manifest_index} "
                f"EPE={report['baseline_metrics']['epe']:.3f}m"
            )
    finally:
        for archive in getattr(dataset, "_archives", {}).values():
            archive.close()

    summary = _aggregate_reports(
        reports,
        output_dir=output_dir,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    run_manifest = {
        "checkpoint": checkpoint,
        "config": (
            Path(args.config).expanduser().resolve()
            if args.config is not None
            else checkpoint_config_path(checkpoint).resolve()
        ),
        "model_path": getattr(module.model, "model_path", None),
        "model_path_override": args.model_path,
        "frames": episode.frames_path,
        "manifest": episode.manifest_path,
        "manifest_source": episode.source,
        "selected_episode_positions": selected_positions,
        "selected_manifest_indices": [episode.record_indices[index] for index in selected_positions],
        "device": str(device),
        "dtype": str(dtype),
        "attention_backend": "eager",
        "ablation_cutoffs": cutoffs,
        "bootstrap": {"seed": args.seed, "resamples": args.bootstrap_samples},
        "history_compressor": {
            "enabled": bool(compressor_config.enabled),
            "mode": str(compressor_config.mode),
            "qformer_status": qformer_status,
        },
        "normalization": {
            "mass": "sum of attention probabilities over each source group",
            "density": (
                "source mass divided by causally available, non-padding source keys "
                "for each query group"
            ),
        },
        "spatial_overlay_geometry": (
            "RGB shown after the same longest-side resize + symmetric padding geometry "
            "used by the model; attention maps are not overlaid on the original camera RGB."
        ),
        "sample_token_layouts": [
            {
                "sample_position": report["sample_position"],
                "manifest_record_index": report["manifest_record_index"],
                "layout": report["token_layout"],
                "frame_attention_layout": report["frame_attention_layout"],
            }
            for report in reports
        ],
        "summary": summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(f"[attention-vis] wrote {output_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
