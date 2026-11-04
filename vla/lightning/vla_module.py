import inspect
import re
from collections.abc import MutableMapping
from typing import Any, Dict, Mapping, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

import json
import os

import numpy as np

from vla.config import (
    AuxiliaryTasksConfig,
    CustomVLAConfig,
    ExecutionStage,
    HistoryCompressorConfig,
    ModelExtensionContext,
    ModelExtensionsConfig,
    SAM2Config,
)
from vla.heads import TrajectoryMLP, compute_auxiliary_losses
from vla.models import CustomVLA, InternVL3VLA, InternVLVLA, Qwen2VLVLA, Qwen3VLVLA
from vla.utils.checkpoint_config import extract_checkpoint_hparams
from vla.utils.trainable_state_dict import (
    require_complete_trainable_state_dict,
)


def _detect_model_type(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        architectures = config.get("architectures", [])
        if model_type == "qwen2_vl" or "Qwen2VLForConditionalGeneration" in architectures:
            return "qwen2vl"
        if model_type == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in architectures:
            return "qwen3vl"
        if "moe" in model_type.lower() or any("moe" in str(name).lower() for name in architectures):
            raise ValueError("Qwen3-VL MoE checkpoints are not supported")
        if model_type == "internvl" or "InternVLModel" in architectures:
            return "internvl3"
        if model_type == "internvl_chat" or "InternVLChatModel" in architectures:
            return "internvl"
    raise ValueError(f"Cannot detect model_type from {config_path}")


class VLAPlModule(pl.LightningModule):
    """PyTorch Lightning module for training the VLA backbones supported here."""

    @staticmethod
    def _serializable_init_hparams(
        init_values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Serialize exactly the constructor parameters Lightning must restore."""
        parameter_names = (
            name
            for name in inspect.signature(VLAPlModule.__init__).parameters
            if name != "self"
        )
        hparams = {name: init_values[name] for name in parameter_names}
        custom_config = CustomVLAConfig.from_mapping(hparams["custom_config"])
        hparams["custom_config"] = custom_config.to_mapping()
        auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            hparams["auxiliary_tasks_config"]
        )
        hparams["auxiliary_tasks_config"] = auxiliary_tasks_config.to_mapping()
        model_extensions = ModelExtensionsConfig.from_values(
            sam2=hparams["sam2_config"],
            history_compressor=hparams["history_compressor_config"],
        )
        hparams.update(model_extensions.architecture_mapping())
        return hparams

    @staticmethod
    def _resolve_trajectory_configuration(
        num_queries: int,
        n_future: int,
    ) -> tuple[int, int]:
        """Validate the query-token/waypoint contract before model creation."""
        resolved_num_queries = int(num_queries)
        if n_future is None:
            raise ValueError("n_future must be supplied by the training data configuration")
        resolved_n_future = int(n_future)
        if resolved_num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {resolved_num_queries}")
        if resolved_n_future <= 0:
            raise ValueError(f"n_future must be positive, got {resolved_n_future}")
        if (
            resolved_num_queries > 1
            and resolved_num_queries != resolved_n_future
        ):
            raise ValueError(
                "num_queries must equal n_future when more than one learnable "
                "query token is used, got "
                f"{resolved_num_queries} and {resolved_n_future}"
            )
        return resolved_num_queries, resolved_n_future

    def __init__(
        self,
        model_path: str,
        n_future: int,
        model_type: Optional[str] = None,
        num_queries: int = 24,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        freeze_mlp1: bool = True,
        freeze_action_head: bool = False,
        action_head_dims: tuple = (512, 2),
        action_head_type: str = "legacy",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer_param_groups: Optional[list] = None,
        adamw_config: Optional[Mapping[str, Any]] = None,
        onecycle_config: Optional[Mapping[str, Any]] = None,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        yaw_weight: float = 1.0,
        x_weight: float = 1.0,
        y_weight: float = 1.0,
        local_files_only: bool = True,
        dtype: Optional[torch.dtype] = None,
        gradient_checkpointing: bool = False,
        attn_implementation: str = "sdpa",
        sam2_config: Optional[Union[SAM2Config, Mapping[str, Any]]] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, Any]]
        ] = None,
        vision_model_path: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        input_size: int = 448,
        custom_config: Optional[
            Union[CustomVLAConfig, Mapping[str, Any]]
        ] = None,
        auxiliary_tasks_config: Optional[
            Union[AuxiliaryTasksConfig, Mapping[str, Any]]
        ] = None,
    ):
        super().__init__()
        model_extensions = ModelExtensionsConfig.from_values(
            sam2=sam2_config,
            history_compressor=history_compressor_config,
        )
        sam2_config = model_extensions.sam2
        history_compressor_config = model_extensions.history_compressor
        custom_config = CustomVLAConfig.from_mapping(custom_config)
        auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            auxiliary_tasks_config
        )
        history_compressor_config.validate_grounding(auxiliary_tasks_config.grounding_enabled)
        action_head_type = TrajectoryMLP.canonicalize_action_head_type(
            action_head_type
        )
        num_queries, n_future = self._resolve_trajectory_configuration(
            num_queries,
            n_future,
        )
        if model_type is None:
            if model_path is None:
                raise ValueError("model_path is required when model_type is not explicit")
            model_type = _detect_model_type(model_path)
        if auxiliary_tasks_config.enabled and model_type not in {"custom", "internvl3"}:
            raise ValueError(
                "grounding/VQA auxiliary tasks require model_type='custom' or 'internvl3'"
            )
        model_extensions.validate_for(
            ModelExtensionContext(model_type=model_type)
        )
        hparams = self._serializable_init_hparams(locals())
        self.save_hyperparameters(hparams)
        self.model_type = model_type
        self.sam2_config = sam2_config
        self.history_compressor_config = history_compressor_config
        self.model_extensions = model_extensions
        self.auxiliary_tasks_config = auxiliary_tasks_config
        self.action_head_type = action_head_type
        self.optimizer_param_groups = optimizer_param_groups or []
        self.adamw_config = dict(adamw_config or {})
        self.onecycle_config = dict(onecycle_config or {})

        if model_type == "custom":
            if not vision_model_path or not llm_model_path:
                raise ValueError(
                    "model_type='custom' requires vision_model_path and llm_model_path"
                )
            self.model = CustomVLA(
                vision_model_path=vision_model_path,
                llm_model_path=llm_model_path,
                input_size=input_size,
                num_queries=num_queries,
                n_future=n_future,
                freeze_vision=freeze_vision,
                freeze_llm=freeze_llm,
                freeze_action_head=freeze_action_head,
                action_head_dims=action_head_dims,
                action_head_type=action_head_type,
                dtype=dtype,
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
                gradient_checkpointing=gradient_checkpointing,
                custom_config=custom_config,
                sam2_config=sam2_config,
                history_compressor_config=history_compressor_config,
                auxiliary_tasks_config=auxiliary_tasks_config,
            )
        elif model_type == "qwen2vl":
            if model_path is None:
                raise ValueError("model_path is required for qwen2vl")
            self.model = Qwen2VLVLA(
                model_path=model_path,
                num_queries=num_queries,
                n_future=n_future,
                freeze_vision=freeze_vision,
                freeze_llm=freeze_llm,
                freeze_action_head=freeze_action_head,
                action_head_dims=action_head_dims,
                action_head_type=action_head_type,
                dtype=dtype,
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
                sam2_config=sam2_config,
            )
        elif model_type == "internvl":
            if model_path is None:
                raise ValueError("model_path is required for internvl")
            self.model = InternVLVLA(
                model_path=model_path,
                num_queries=num_queries,
                n_future=n_future,
                freeze_vision=freeze_vision,
                freeze_llm=freeze_llm,
                freeze_mlp1=freeze_mlp1,
                action_head_dims=action_head_dims,
                action_head_type=action_head_type,
                dtype=dtype,
                local_files_only=local_files_only,
                gradient_checkpointing=gradient_checkpointing,
                attn_implementation=attn_implementation,
                sam2_config=sam2_config,
            )
        elif model_type == "internvl3":
            if model_path is None:
                raise ValueError("model_path is required for internvl3")
            self.model = InternVL3VLA(
                model_path=model_path,
                num_queries=num_queries,
                n_future=n_future,
                freeze_vision=freeze_vision,
                freeze_llm=freeze_llm,
                freeze_mlp1=freeze_mlp1,
                action_head_dims=action_head_dims,
                action_head_type=action_head_type,
                dtype=dtype,
                local_files_only=local_files_only,
                gradient_checkpointing=gradient_checkpointing,
                attn_implementation=attn_implementation,
                sam2_config=sam2_config,
                history_compressor_config=history_compressor_config,
                auxiliary_tasks_config=auxiliary_tasks_config,
            )
        elif model_type == "qwen3vl":
            if model_path is None:
                raise ValueError("model_path is required for qwen3vl")
            self.model = Qwen3VLVLA(
                model_path=model_path,
                num_queries=num_queries,
                n_future=n_future,
                freeze_vision=freeze_vision,
                freeze_llm=freeze_llm,
                freeze_action_head=freeze_action_head,
                action_head_dims=action_head_dims,
                action_head_type=action_head_type,
                dtype=dtype,
                local_files_only=local_files_only,
                gradient_checkpointing=gradient_checkpointing,
                attn_implementation=attn_implementation,
                sam2_config=sam2_config,
                history_compressor_config=history_compressor_config,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.num_queries = self.model.num_queries
        self.n_future = self.model.n_future
        self.num_image_tokens = getattr(self.model, "image_seq_length", None)

        # Metrics.
        self.train_loss = []
        self.val_ade = []
        self.val_fde = []

    def _extension_context(
        self,
        stage: ExecutionStage,
    ) -> ModelExtensionContext:
        return ModelExtensionContext(
            stage=stage,
            model_type=self.model_type,
        )

    def _validate_extension_batch(
        self,
        batch: Mapping[str, Any],
        stage: ExecutionStage,
    ) -> None:
        """Enforce stage-specific extension inputs at the Lightning boundary."""
        if not isinstance(batch, Mapping):
            raise TypeError("VLA batch must be a mapping")
        if not isinstance(stage, ExecutionStage):
            raise TypeError("stage must be an ExecutionStage")

        requirements = self.model_extensions.validate_for(
            self._extension_context(stage)
        )
        has_history_masks = batch.get("history_masks") is not None
        has_history_valid = batch.get("history_mask_valid") is not None
        if has_history_masks != has_history_valid:
            raise ValueError(
                "history_masks and history_mask_valid must be supplied together"
            )

        has_boxes = batch.get("history_boxes") is not None
        has_box_valid = batch.get("history_box_valid") is not None
        if has_boxes != has_box_valid:
            raise ValueError("history_boxes and history_box_valid must be supplied together")
        required_fields = []
        if requirements.history_boxes:
            required_fields.extend(("history_boxes", "history_box_valid"))
        if requirements.current_gt_masks:
            required_fields.append("masks")
        if requirements.history_masks:
            required_fields.extend(
                ("history_masks", "history_mask_valid")
            )
        if (
            self.auxiliary_tasks_config.enabled
            and stage in {ExecutionStage.TRAIN, ExecutionStage.OFFLINE_EVAL}
        ):
            required_fields.extend(
                (
                    "grounding_bbox_target",
                    "grounding_visibility_target",
                    "vqa_labels",
                )
            )
        missing = [
            field_name
            for field_name in required_fields
            if batch.get(field_name) is None
        ]
        if missing:
            raise ValueError(
                f"{stage.value} batch is missing model-extension input(s): "
                + ", ".join(missing)
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Validate training-only extension combinations for direct Lightning use."""
        if stage == "fit":
            self.model_extensions.validate_for(
                self._extension_context(ExecutionStage.TRAIN)
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        extension_stage: ExecutionStage = ExecutionStage.MODEL,
    ) -> Dict[str, torch.Tensor]:
        self._validate_extension_batch(batch, extension_stage)
        kwargs = {
            "pixel_values": batch["pixel_values"],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "trajectories": batch.get("trajectories"),
            "masks": batch.get("masks"),
        }
        if "history_masks" in batch:
            kwargs["history_masks"] = batch["history_masks"]
        if "history_mask_valid" in batch:
            kwargs["history_mask_valid"] = batch["history_mask_valid"]
        for key in ("history_boxes", "history_box_valid", "current_frame_qformer"):
            if key in batch:
                kwargs[key] = batch[key]
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"]
        if "mm_token_type_ids" in batch:
            kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"]
        return self.model(**kwargs)

    def _compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.

        For (x, y, yaw) trajectories, each dimension is weighted separately so
        that lateral motion (y) and heading (yaw) are not drowned out by the
        larger-scale forward motion (x).
        """
        traj_dim = gt.shape[-1]
        if traj_dim == 3:
            x_loss = torch.nn.functional.mse_loss(pred[..., 0], gt[..., 0])
            y_loss = torch.nn.functional.mse_loss(pred[..., 1], gt[..., 1])
            yaw_loss = torch.nn.functional.mse_loss(pred[..., 2], gt[..., 2])
            loss = (
                self.hparams.get("x_weight", 1.0) * x_loss
                + self.hparams.get("y_weight", 1.0) * y_loss
                + self.hparams.get("yaw_weight", 1.0) * yaw_loss
            )
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            batch,
            extension_stage=ExecutionStage.TRAIN,
        )
        # Re-compute weighted loss to handle (x,y,yaw) properly.
        trajectory_loss = self._compute_loss(outputs["pred_traj"], batch["trajectories"])
        loss = trajectory_loss
        mask_loss = outputs.get("mask_loss")
        if self.sam2_config.enabled:
            if mask_loss is None:
                raise ValueError(
                    "model.sam2_config.enabled requires a `masks` tensor in every training batch"
                )
            loss = loss + self.sam2_config.loss_weight * mask_loss
        auxiliary_losses = compute_auxiliary_losses(
            outputs, batch, self.auxiliary_tasks_config
        )
        if self.auxiliary_tasks_config.grounding_enabled:
            loss = loss + self.auxiliary_tasks_config.grounding_loss_weight * auxiliary_losses["grounding"]
        if self.auxiliary_tasks_config.vqa_enabled:
            loss = loss + self.auxiliary_tasks_config.vqa_loss_weight * auxiliary_losses["vqa"]
        batch_size = batch["trajectories"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        if self.sam2_config.enabled:
            self.log("train/trajectory_loss", trajectory_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("train/mask_loss", mask_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("train/mask_iou", outputs["mask_iou"], on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in auxiliary_losses.items():
            self.log(f"train/{name}_loss", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            batch,
            extension_stage=ExecutionStage.OFFLINE_EVAL,
        )
        pred = outputs["pred_traj"]  # (B, T, C)
        gt = batch["trajectories"]  # (B, T, C)
        batch_size = gt.shape[0]

        trajectory_loss = self._compute_loss(pred, gt)
        loss = trajectory_loss
        mask_loss = outputs.get("mask_loss")
        if self.sam2_config.enabled:
            if mask_loss is None:
                raise ValueError(
                    "model.sam2_config.enabled requires a `masks` tensor in every validation batch"
                )
            loss = loss + self.sam2_config.loss_weight * mask_loss
        auxiliary_losses = compute_auxiliary_losses(
            outputs, batch, self.auxiliary_tasks_config
        )
        if self.auxiliary_tasks_config.grounding_enabled:
            loss = loss + self.auxiliary_tasks_config.grounding_loss_weight * auxiliary_losses["grounding"]
        if self.auxiliary_tasks_config.vqa_enabled:
            loss = loss + self.auxiliary_tasks_config.vqa_loss_weight * auxiliary_losses["vqa"]

        # Euclidean distance per point (x, y only).
        dists = torch.norm(pred[..., :2] - gt[..., :2], dim=-1)  # (B, T)
        ade = dists.mean()
        fde = dists[:, -1].mean()

        log_dict = {
            "val/loss": loss,
            "val/ade": ade,
            "val/fde": fde,
        }

        # Yaw error (radians) if available.
        if gt.shape[-1] == 3:
            yaw_diff = pred[..., 2] - gt[..., 2]
            yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
            yaw_error = yaw_diff.abs().mean()
            yaw_fde = yaw_diff[:, -1].abs().mean()
            log_dict["val/yaw_error"] = yaw_error
            log_dict["val/yaw_fde"] = yaw_fde

        if self.sam2_config.enabled:
            log_dict["val/trajectory_loss"] = trajectory_loss
            log_dict["val/mask_loss"] = mask_loss
            log_dict["val/mask_iou"] = outputs["mask_iou"]
        for name, value in auxiliary_losses.items():
            log_dict[f"val/{name}_loss"] = value

        for k, v in log_dict.items():
            self.log(k, v, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return log_dict

    def configure_optimizers(self):
        # Build optional MH4/Gitee regex parameter groups. Patterns are matched
        # against names inside self.model (without the Lightning ``model.`` prefix).
        groups = []
        assigned = set()
        for spec in self.optimizer_param_groups:
            name = str(spec.get("name", f"group{len(groups)}"))
            patterns = [re.compile(str(pattern)) for pattern in spec.get("patterns", [])]
            matched = []
            for param_name, parameter in self.model.named_parameters():
                if not parameter.requires_grad or param_name in assigned:
                    continue
                if any(pattern.search(param_name) for pattern in patterns):
                    matched.append(parameter)
                    assigned.add(param_name)
            if matched:
                groups.append({"params": matched, "lr": float(spec.get("lr", self.hparams.lr))})
        remaining = [
            parameter for param_name, parameter in self.model.named_parameters()
            if parameter.requires_grad and param_name not in assigned
        ]
        if remaining:
            groups.append({"params": remaining, "lr": self.hparams.lr})
        if not groups:
            raise RuntimeError("No trainable parameters remain for the optimizer")

        adamw = dict(self.adamw_config)
        if "betas" in adamw:
            adamw["betas"] = tuple(adamw["betas"])
        adamw.setdefault("weight_decay", self.hparams.weight_decay)
        optimizer = AdamW(groups, **adamw)

        # Support epoch-based training: when max_steps is negative / unset, derive
        # total optimization steps from the trainer's estimate.
        max_steps = self.hparams.max_steps
        if max_steps is None or max_steps <= 0:
            if self.trainer is not None and hasattr(self.trainer, "estimated_stepping_batches"):
                max_steps = int(self.trainer.estimated_stepping_batches)
            else:
                # Fallback: train for a large number of steps; scheduler will be
                # sub-optimal but training will still respect max_epochs.
                max_steps = 1_000_000

        onecycle = dict(self.onecycle_config)
        onecycle.setdefault("cycle_momentum", False)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in groups],
            total_steps=max_steps,
            pct_start=min(0.3, self.hparams.warmup_steps / max(1, max_steps)),
            anneal_strategy="cos",
            **onecycle,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Keep checkpoints lightweight by dropping frozen backbone weights."""
        if self.model_type == "custom":
            checkpoint["custom_vla_structure"] = self.model.structure_mapping()
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if k in trainable_keys
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore omitted frozen state for ordinary lightweight checkpoints."""
        if self.model_type == "custom":
            saved_structure = checkpoint.get("custom_vla_structure")
            if saved_structure is None:
                raise RuntimeError(
                    "Custom VLA checkpoint has no resolved structure metadata; "
                    "it cannot be restored safely"
                )
            try:
                self.model.assert_checkpoint_structure(saved_structure)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(str(exc)) from exc
        checkpoint_hparams = extract_checkpoint_hparams(checkpoint)
        current_extensions = getattr(self, "model_extensions", None)
        if current_extensions is None:
            current_extensions = ModelExtensionsConfig.from_values(
                sam2=getattr(self, "sam2_config", None),
                history_compressor=getattr(
                    self,
                    "history_compressor_config",
                    None,
                ),
            )
        try:
            current_extensions.assert_checkpoint_compatible(
                checkpoint_hparams
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Checkpoint model-extension architecture does not match the "
                f"constructed model: {exc}"
            ) from exc

        state_dict = checkpoint.get("state_dict")
        # DeepSpeed restores the model separately and invokes this hook with
        # its Lightning client state, which deliberately has no state_dict.
        if state_dict is None:
            if "ds_config" in checkpoint:
                return
            raise KeyError("Checkpoint has no state_dict.")

        if not isinstance(state_dict, MutableMapping):
            raise TypeError("Checkpoint state_dict must be a mutable mapping.")

        current_state = self.state_dict()
        missing_keys = [key for key in current_state if key not in state_dict]
        require_complete_trainable_state_dict(self, state_dict)

        for key in missing_keys:
            state_dict[key] = current_state[key]
