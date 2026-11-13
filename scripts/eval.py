#!/usr/bin/env python3
"""Evaluate a trained VLA checkpoint on its configured dataset.

Examples:
    python scripts/eval.py \
        --checkpoint outputs/.../checkpoints/last.ckpt

    # Override config.yaml next to the checkpoint.
    python scripts/eval.py \
        --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml \
        --checkpoint outputs/.../checkpoints/last.ckpt
"""
import argparse
import contextlib
import json
import os
import sys

# Allow running this script from the project root or from scripts/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from vla.config import (
    ExecutionStage,
    ModelExtensionContext,
    ModelExtensionsConfig,
)
from vla.data import DobotVLADataModule, TrackVLADataModule
from vla.lightning import VLAPlModule
from vla.utils.checkpoint_config import (
    checkpoint_config_path,
    load_checkpoint_hparams,
    load_checkpoint_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML override. Defaults to config.yaml beside the checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--training_json_list", type=str, default=None,
                        help="Override the training_json_list from config (e.g. local CPFS path).")
    parser.add_argument("--trackvla_sources", type=str, default=None,
                        help="Override comma-separated TrackVLA source paths from config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_checkpoint_config(args.checkpoint, args.config)
    if args.config is None:
        print(f"[eval] Loaded config from {checkpoint_config_path(args.checkpoint)}")
    else:
        print(f"[eval] Loaded config override from {args.config}")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.checkpoint)),
            "eval_results",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    dtype_str = config["model"].get("dtype", "bf16")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    pl.seed_everything(config.get("data", {}).get("seed", 42), workers=True)
    torch.set_float32_matmul_precision("high")

    checkpoint_hparams = load_checkpoint_hparams(args.checkpoint)
    dataset_type = config["data"].get("dataset_type", "nuscenes")
    model_extensions = ModelExtensionsConfig.resolve_checkpoint(
        model_config=config["model"],
        checkpoint_hparams=checkpoint_hparams,
        context=ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type=dataset_type,
            model_type=(
                config["model"].get("model_type")
                or checkpoint_hparams.get("model_type")
            ),
            dense_tokens_per_frame=config["data"].get(
                "num_image_tokens",
                256,
            ),
        ),
    )
    model_extensions.apply_to_model_mapping(config["model"])
    extension_requirements = model_extensions.validate_for(
        ModelExtensionContext(
            stage=ExecutionStage.OFFLINE_EVAL,
            dataset_type=dataset_type,
            model_type=(
                config["model"].get("model_type")
                or checkpoint_hparams.get("model_type")
            ),
            dense_tokens_per_frame=config["data"].get(
                "num_image_tokens",
                256,
            ),
        )
    )
    sam2_config = model_extensions.sam2
    history_compressor_config = model_extensions.history_compressor

    # Load module from checkpoint.
    print(f"[eval] Loading checkpoint {args.checkpoint} ...")
    module = VLAPlModule.load_from_checkpoint(
        args.checkpoint,
        model_path=config["model"].get("model_path"),
        model_type=config["model"].get("model_type", None),
        vision_model_path=config["model"].get("vision_model_path"),
        llm_model_path=config["model"].get("llm_model_path"),
        input_size=config["data"].get("input_size", 448),
        custom_config=config["model"].get("custom"),
        num_queries=config["model"]["num_queries"],
        n_future=config["data"]["n_future"],
        freeze_vision=config["model"]["freeze_vision"],
        freeze_llm=config["model"]["freeze_llm"],
        freeze_mlp1=config["model"].get("freeze_mlp1", True),
        freeze_action_head=config["model"].get("freeze_action_head", False),
        action_head_dims=tuple(config["model"]["action_head_dims"]),
        action_head_type=config["model"].get("action_head_type", "legacy"),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        yaw_weight=config["training"].get("yaw_weight", 1.0),
        x_weight=config["training"].get("x_weight", 1.0),
        y_weight=config["training"].get("y_weight", 1.0),
        dtype=dtype,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        sam2_config=sam2_config,
        history_compressor_config=history_compressor_config,
        auxiliary_tasks_config=config["model"].get("auxiliary_tasks_config", checkpoint_hparams.get("auxiliary_tasks_config")),
    )
    if module.model_type == "custom":
        inferred_image_tokens = int(module.num_image_tokens)
        configured_image_tokens = config["data"].get("num_image_tokens")
        if configured_image_tokens is not None and int(configured_image_tokens) != inferred_image_tokens:
            raise ValueError(
                "data.num_image_tokens does not match the custom ViT/input_size: "
                f"configured {configured_image_tokens}, inferred {inferred_image_tokens}"
            )
        config["data"]["num_image_tokens"] = inferred_image_tokens
    module.eval()
    module.freeze()

    # Data module.
    if dataset_type == "dobot":
        training_json_list = args.training_json_list or config["data"]["training_json_list"]
        dm_kwargs = {
            "training_json_list": training_json_list,
            "tokenizer": module.model.tokenizer,
            "processor": getattr(module.model, "image_processor", None),
            "model_type": module.model_type,
            "image_token": getattr(module.model, "image_token", None),
            "input_size": config["data"]["input_size"],
            "num_history_frames": config["data"]["num_history_frames"],
            "history_duration": config["data"].get("history_duration", 1.0),
            "n_future": config["data"]["n_future"],
            "future_dt": config["data"].get("future_dt", 0.25),
            "cam_channel": config["data"]["cam_channel"],
            "batch_size": args.batch_size or config["data"]["batch_size"],
            "num_workers": args.num_workers,
            "val_split": config["data"]["val_split"],
            "sample_interval": config["data"].get("sample_interval", 5),
            "prompt": config["data"]["prompt"],
            "num_image_tokens": config["data"].get("num_image_tokens", 256),
            "max_length": config["data"].get("max_length", 2048),
            "cache_dir": config["data"].get("cache_dir"),
            "seed": config["data"].get("seed", 42),
            "use_scene_desc": config["data"].get("use_scene_desc", True),
            "scene_desc_augment": config["data"].get("scene_desc_augment", True),
        }
        dm = DobotVLADataModule(**dm_kwargs)
    elif dataset_type == "trackvla":
        source_paths = (
            args.trackvla_sources
            or config["data"].get("trackvla_sources")
            or config["data"].get("train_json")
        )
        if not source_paths:
            raise ValueError(
                "TrackVLA requires data.trackvla_sources (or the legacy data.train_json)"
            )
        dm_kwargs = {
            "source_paths": source_paths,
            "tokenizer": module.model.tokenizer,
            "input_size": config["data"]["input_size"],
            "num_history_frames": config["data"]["num_history_frames"],
            "history_frame_interval": config["data"].get("history_frame_interval", 0),
            "n_future": config["data"]["n_future"],
            "trajectory_dim": config["data"].get("trajectory_dim", 3),
            "batch_size": args.batch_size or config["data"]["batch_size"],
            "num_workers": args.num_workers,
            "val_split": config["data"]["val_split"],
            "prompt": config["data"].get(
                "prompt",
                "Predict the follower's future trajectory in the ego coordinate frame.",
            ),
            "num_image_tokens": config["data"].get("num_image_tokens", 256),
            "max_length": config["data"].get("max_length", 1200),
            "seed": config["data"].get("seed", 42),
            "model_type": module.model_type,
            "image_token": getattr(module.model, "image_token", None),
            "qwen_min_pixels": config["data"].get("qwen_min_pixels", 3136),
            "qwen_max_pixels": config["data"].get("qwen_max_pixels", 100_000),
            "index_cache_dir": config["data"].get("index_cache_dir"),
            "max_open_archives": config["data"].get("max_open_archives", 32),
            "load_masks": extension_requirements.current_gt_masks,
            "history_compressor_config": history_compressor_config,
            "auxiliary_tasks_config": module.auxiliary_tasks_config,
            "history_mask_augmentation": False,
        }
        if module.model_type in {"qwen2vl", "qwen3vl"}:
            dm_kwargs["processor"] = module.model.tokenizer
        elif module.model_type == "custom":
            dm_kwargs["processor"] = module.model.image_processor
        dm = TrackVLADataModule(**dm_kwargs)
    else:
        from vla.data import NuScenesVLADataModule

        dm_kwargs = {
            "dataroot": config["data"]["dataroot"],
            "version": config["data"]["version"],
            "tokenizer": module.model.tokenizer,
            "input_size": config["data"]["input_size"],
            "num_history_frames": config["data"]["num_history_frames"],
            "n_future": config["data"]["n_future"],
            "dt": config["data"]["dt"],
            "cam_channel": config["data"]["cam_channel"],
            "batch_size": args.batch_size or config["data"]["batch_size"],
            "num_workers": args.num_workers,
            "val_split": config["data"]["val_split"],
            "prompt": config["data"]["prompt"],
            "cache_dir": config["data"].get("cache_dir"),
            "num_image_tokens": config["data"].get("num_image_tokens", 256),
            "max_length": config["data"].get("max_length", 2048),
            "model_type": module.model_type,
            "image_token": getattr(module.model, "image_token", None),
            "qwen_min_pixels": config["data"].get("qwen_min_pixels", 3136),
            "qwen_max_pixels": config["data"].get("qwen_max_pixels", 100_000),
        }
        if module.model_type == "qwen2vl":
            dm_kwargs["processor"] = module.model.tokenizer
        elif module.model_type == "custom":
            dm_kwargs["processor"] = module.model.image_processor
        dm = NuScenesVLADataModule(**dm_kwargs)
    dm.setup()

    dataloader = dm.val_dataloader()
    device = module.device

    all_preds = []
    all_gts = []
    all_tokens = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for k in [
                "pixel_values",
                "input_ids",
                "attention_mask",
                "trajectories",
                "masks",
                "history_masks",
                "history_mask_valid",
                "history_boxes",
                "history_box_valid",
                "image_grid_thw",
                "mm_token_type_ids",
            ]:
                if k in batch and hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)
            autocast_ctx = torch.amp.autocast(device.type, dtype=dtype) if device.type != "cpu" else contextlib.nullcontext()
            with autocast_ctx:
                outputs = module(
                    batch,
                    extension_stage=ExecutionStage.OFFLINE_EVAL,
                )
            pred = outputs["pred_traj"].float().cpu().numpy()
            gt = batch["trajectories"].float().cpu().numpy()
            all_preds.append(pred)
            all_gts.append(gt)
            all_tokens.extend(batch["tokens"])
            if (batch_idx + 1) % 50 == 0:
                print(f"[eval] Processed {batch_idx + 1} / {len(dataloader)} batches")

    preds = np.concatenate(all_preds, axis=0)  # (N, T, C)
    gts = np.concatenate(all_gts, axis=0)  # (N, T, C)

    # Position metrics (x, y).
    dists = np.linalg.norm(preds[..., :2] - gts[..., :2], axis=-1)  # (N, T)
    ade = dists.mean()
    fde = dists[:, -1].mean()
    ade_per_step = dists.mean(axis=0)
    fde_per_sample = dists[:, -1]

    metrics = {
        "ADE": float(ade),
        "FDE": float(fde),
        "ADE_per_step": [float(x) for x in ade_per_step],
        "FDE_per_sample_mean": float(fde_per_sample.mean()),
        "FDE_per_sample_std": float(fde_per_sample.std()),
        "num_samples": int(preds.shape[0]),
    }
    if (
        dataset_type == "trackvla"
        and history_compressor_config.requires_history_masks
    ):
        metrics["history_mask_source"] = "clean_gt_oracle"
    if dataset_type == "trackvla" and history_compressor_config.requires_history_boxes:
        metrics["history_box_source"] = "clean_gt_oracle"

    # Yaw metrics if available.
    if gts.shape[-1] == 3:
        yaw_diff = preds[..., 2] - gts[..., 2]
        yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
        yaw_error = np.abs(yaw_diff).mean()
        yaw_fde = np.abs(yaw_diff[:, -1]).mean()
        metrics["YawError"] = float(yaw_error)
        metrics["YawFDE"] = float(yaw_fde)
        metrics["YawError_per_step"] = [float(x) for x in np.abs(yaw_diff).mean(axis=0)]

    print("\n[eval] Metrics:")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"  {k}: {v[:5]} ... (len={len(v)})")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results.
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        os.path.join(args.output_dir, "predictions.npz"),
        predictions=preds,
        ground_truth=gts,
        tokens=all_tokens,
    )
    print(f"[eval] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
