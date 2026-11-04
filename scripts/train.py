#!/usr/bin/env python3
"""Train DobotTrack with PyTorch Lightning.

Supports single-node and multi-node DDP out of the box via PyTorch Lightning's
strategy argument.

Examples:
    # Single-GPU
    python scripts/train.py --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml

    # Multi-GPU single node
    python scripts/train.py --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml \
        trainer.devices=4

    # Multi-node DDP (2 nodes, 4 GPUs each)
    # Node 0:
    python scripts/train.py --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml \
        trainer.devices=4 trainer.num_nodes=2 trainer.strategy=ddp
    # Node 1:
    python scripts/train.py --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml \
        trainer.devices=4 trainer.num_nodes=2 trainer.strategy=ddp
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running this script from the project root or from scripts/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from vla.config import (
    AuxiliaryTasksConfig,
    ExecutionStage,
    ModelExtensionContext,
    ModelExtensionsConfig,
)
from vla.lightning import VLAPlModule
from vla.utils.checkpoint_config import (
    load_checkpoint_hparams,
    save_checkpoint_config,
)


class PersistentProgressCallback(Callback):
    """Print newline-delimited progress that remains visible in redirected logs."""

    def __init__(self, every_n_batches: int = 10):
        super().__init__()
        self.every_n_batches = max(1, int(every_n_batches))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(
                f"[progress] epoch={trainer.current_epoch + 1}/"
                f"{trainer.max_epochs} train_batches={trainer.num_training_batches}",
                flush=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if (batch_idx + 1) % self.every_n_batches != 0:
            return
        total = trainer.num_training_batches
        percent = 100.0 * (batch_idx + 1) / total if total else 0.0
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        elif torch.is_tensor(outputs):
            loss = outputs
        loss_text = ""
        if loss is not None:
            loss_text = f" loss={float(loss.detach().cpu()):.6f}"
        print(
            f"[progress] epoch={trainer.current_epoch + 1}/"
            f"{trainer.max_epochs} batch={batch_idx + 1}/{total} "
            f"({percent:.2f}%) global_step={trainer.global_step}{loss_text}",
            flush=True,
        )


def preflight_acceleration_dependencies(config: Dict[str, Any]) -> None:
    """Fail early when an explicitly requested optional accelerator is absent."""
    model_config = config.get("model", {})
    trainer_config = config.get("trainer", {})
    strategy = str(trainer_config.get("strategy", "")).lower()
    attn_implementation = str(
        model_config.get("attn_implementation", "sdpa")
    ).lower()

    missing = []
    if "deepspeed" in strategy and importlib.util.find_spec("deepspeed") is None:
        missing.append("deepspeed (required by trainer.strategy)")
    if (
        attn_implementation == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        missing.append("flash-attn (required by model.attn_implementation)")

    if missing:
        raise RuntimeError(
            "Requested optional acceleration dependencies are not installed: "
            + ", ".join(missing)
        )


def load_config(config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must decode to a mapping")
    if not isinstance(config.get("model"), dict):
        raise ValueError(f"Config {config_path} must contain a model mapping")
    if not isinstance(config.get("data"), dict):
        raise ValueError(f"Config {config_path} must contain a data mapping")
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            child = node.get(k)
            if child is None:
                child = {}
                node[k] = child
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot apply nested override {key_path}: {k} is not a mapping"
                )
            node = child
        old = node.get(keys[-1])
        try:
            if isinstance(old, str):
                parsed_value = value
            elif isinstance(old, float):
                parsed_value = float(value)
            else:
                parsed_value = yaml.safe_load(value)
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Could not parse override {key_path}={value!r}: {exc}"
            ) from exc
        node[keys[-1]] = parsed_value
    return config


def parse_overrides(override_list):
    overrides = {}
    if override_list is None:
        return overrides
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path.")
    parser.add_argument("override", nargs="*", help="Config overrides as key=value.")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    preflight_acceleration_dependencies(config)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Resolve dtype.
    dtype_str = config["model"].get("dtype", "bf16")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    pl.seed_everything(config.get("data", {}).get("seed", 42), workers=True)
    torch.set_float32_matmul_precision("high")

    extension_context = ModelExtensionContext(
        stage=ExecutionStage.TRAIN,
        dataset_type=config["data"].get("dataset_type", "nuscenes"),
        model_type=config["model"].get("model_type"),
        dense_tokens_per_frame=config["data"].get(
            "num_image_tokens",
            256,
        ),
    )
    if args.resume is not None:
        model_extensions = ModelExtensionsConfig.resolve_checkpoint(
            model_config=config["model"],
            checkpoint_hparams=load_checkpoint_hparams(args.resume),
            context=extension_context,
        )
    else:
        model_extensions = ModelExtensionsConfig.from_model_mapping(
            config["model"]
        )
    extension_requirements = model_extensions.validate_for(extension_context)
    model_extensions.apply_to_model_mapping(config["model"])
    sam2_config = model_extensions.sam2
    history_compressor_config = model_extensions.history_compressor
    auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
        config["model"].get("auxiliary_tasks_config")
    )
    config["model"]["auxiliary_tasks_config"] = auxiliary_tasks_config.to_mapping()
    dataset_type = config["data"].get("dataset_type", "nuscenes")

    # Lightning module (loads tokenizer/processor as part of the model).
    print("[train] Building Lightning module...")
    module = VLAPlModule(
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
        optimizer_param_groups=config["training"].get("optimizer_param_groups"),
        adamw_config=config["training"].get("adamw"),
        onecycle_config=config["training"].get("onecycle"),
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        yaw_weight=config["training"].get("yaw_weight", 1.0),
        x_weight=config["training"].get("x_weight", 1.0),
        y_weight=config["training"].get("y_weight", 1.0),
        dtype=dtype,
        gradient_checkpointing=config["model"].get("gradient_checkpointing", False),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        sam2_config=sam2_config,
        history_compressor_config=history_compressor_config,
        auxiliary_tasks_config=auxiliary_tasks_config,
    )
    config["model"]["model_type"] = module.model_type
    if module.model_type == "custom":
        inferred_image_tokens = int(module.num_image_tokens)
        configured_image_tokens = config["data"].get("num_image_tokens")
        if configured_image_tokens is not None and int(configured_image_tokens) != inferred_image_tokens:
            raise ValueError(
                "data.num_image_tokens does not match the custom ViT/input_size: "
                f"configured {configured_image_tokens}, inferred {inferred_image_tokens}"
            )
        config["data"]["num_image_tokens"] = inferred_image_tokens
    # Re-run model support validation after automatic model-type detection.
    extension_requirements = model_extensions.validate_for(
        ModelExtensionContext(
            stage=ExecutionStage.TRAIN,
            dataset_type=dataset_type,
            model_type=module.model_type,
            dense_tokens_per_frame=config["data"].get(
                "num_image_tokens",
                256,
            ),
        )
    )
    config_yaml = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    config_path = save_checkpoint_config(checkpoint_dir, config_yaml)
    print(f"[train] Resolved config path: {config_path}")

    # Data module.
    print("[train] Setting up data module...")
    if dataset_type == "dobot":
        from vla.data import DobotVLADataModule
        dm_kwargs = {
            "training_json_list": config["data"]["training_json_list"],
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
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "val_split": config["data"]["val_split"],
            "sample_interval": config["data"].get("sample_interval", 5),
            "prompt": config["data"]["prompt"],
            "num_image_tokens": config["data"].get("num_image_tokens", 256),
            "max_length": config["data"].get("max_length", 2048),
            "cache_dir": config["data"].get("cache_dir"),
            "seed": config["data"].get("seed", 42),
            "num_build_workers": config["data"].get("num_build_workers", 1),
            "use_scene_desc": config["data"].get("use_scene_desc", True),
            "scene_desc_augment": config["data"].get("scene_desc_augment", True),
        }
        dm = DobotVLADataModule(**dm_kwargs)
    elif dataset_type == "trackvla":
        from vla.data import TrackVLADataModule
        source_paths = (
            config["data"].get("trackvla_sources")
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
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
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
            "auxiliary_tasks_config": auxiliary_tasks_config,
            "history_compressor_config": history_compressor_config,
            "history_mask_augmentation": config["data"].get(
                "history_mask_augmentation"
            ),
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
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
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

    # Callbacks.
    checkpointing_enabled = config.get("trainer", {}).get(
        "enable_checkpointing", True
    )
    callbacks = []
    callbacks.append(
        PersistentProgressCallback(
            every_n_batches=int(config["training"].get("log_every_n_steps", 10))
        )
    )
    if checkpointing_enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="vla-{step:06d}-{val/ade:.3f}",
                save_top_k=config.get("trainer", {}).get("save_top_k", 3),
                monitor="val/ade",
                mode="min",
                save_last=config.get("trainer", {}).get("save_last", True),
            )
        )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Logger.
    logger = TensorBoardLogger(
        save_dir=config["output_dir"],
        name="logs",
        version="",
    )

    # Trainer.
    trainer_kwargs = config.get("trainer", {})
    trainer = pl.Trainer(
        accelerator=trainer_kwargs.get("accelerator", "gpu"),
        devices=trainer_kwargs.get("devices", "auto"),
        num_nodes=trainer_kwargs.get("num_nodes", 1),
        strategy=trainer_kwargs.get("strategy", "ddp_find_unused_parameters_true"),
        precision=trainer_kwargs.get("precision", "bf16-mixed"),
        max_epochs=trainer_kwargs.get("max_epochs", -1),
        max_steps=config["training"]["max_steps"],
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        val_check_interval=config["training"].get("val_check_interval", 500),
        check_val_every_n_epoch=trainer_kwargs.get("check_val_every_n_epoch", 1),
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
        enable_checkpointing=checkpointing_enabled,
        callbacks=callbacks,
        logger=logger,
    )

    print(f"[train] Output directory: {config['output_dir']}")
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
