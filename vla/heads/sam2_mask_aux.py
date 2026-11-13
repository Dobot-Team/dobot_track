"""Stateless SAM2 mask auxiliary head for VLA training.

The head intentionally keeps only SAM2's mask decoder.  VLA visual features
replace SAM2 image-encoder features and a language hidden state replaces the
prompt embedding, so temporal SAM2 memory is neither built nor consumed.
"""

from __future__ import annotations

import importlib.machinery
import sys
import threading
import types
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config.sam2 import SAM2Config


def _dice_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _binary_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predicted = logits > 0
    intersection = (predicted * targets.bool()).sum(dim=(2, 3))
    union = predicted.sum(dim=(2, 3)) + targets.bool().sum(dim=(2, 3)) - intersection
    return intersection.float() / (union.float() + 1e-6)


class SAM2MaskAuxiliaryHead(nn.Module):
    """Decode one current-frame mask per VLA sample without temporal memory.

    A complete SAM2 inference model is built directly from the bundled YAML so
    this loader never initializes Hydra's process-global configuration. After
    loading the pretrained weights, this module caches the prompt encoder's
    constant tensors and removes every component used for image encoding or
    video memory. The resulting forward path calls only ``sam_mask_decoder``.
    """

    _INFERENCE_CONFIG_DIRECTORIES = frozenset({"sam2", "sam2.1"})
    _DYNAMIC_MULTIMASK_DEFAULTS = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }
    _SAM2_NAMESPACE_LOCK = threading.RLock()

    def __init__(
        self,
        *,
        sam2_config: SAM2Config,
        image_feature_dim: int,
        language_feature_dim: int,
    ) -> None:
        super().__init__()
        sam2_config.validate()
        if not sam2_config.enabled:
            raise ValueError("SAM2MaskAuxiliaryHead requires an enabled SAM2Config")
        source_root = self._bundled_source_root()

        self.feature_size = sam2_config.feature_size
        self.sam_model = self._build_sam_model(
            config_file=sam2_config.config_file,
            checkpoint=sam2_config.checkpoint,
            source_root=source_root,
        )
        prompt_encoder = getattr(self.sam_model, "sam_prompt_encoder", None)
        mask_decoder = getattr(self.sam_model, "sam_mask_decoder", None)
        if prompt_encoder is None or mask_decoder is None:
            raise RuntimeError("The supplied SAM2 checkpoint does not expose prompt and mask decoders")
        mask_decoder.requires_grad_(sam2_config.train_decoder)
        self.use_high_res_features = self._configure_high_res_features(
            sam_model=self.sam_model,
            decoder=mask_decoder,
            requested=sam2_config.use_high_res_features,
        )

        # Cache the two constant prompt tensors before deleting the prompt
        # encoder. No prompts, image encoder, memory encoder, or memory bank
        # are used during VLA training.
        with torch.no_grad():
            dense_pe = prompt_encoder.get_dense_pe()
            _, empty_dense = prompt_encoder(points=None, boxes=None, masks=None)
        self.register_buffer(
            "sam_pe",
            F.interpolate(dense_pe, size=(self.feature_size, self.feature_size), mode="bilinear", align_corners=False),
            persistent=False,
        )
        self.register_buffer(
            "empty_dense",
            F.interpolate(empty_dense, size=(self.feature_size, self.feature_size), mode="bilinear", align_corners=False),
            persistent=False,
        )

        self._removed_sam_components = []
        for component_name in (
            "image_encoder",
            "memory_encoder",
            "memory_attention",
            "sam_prompt_encoder",
            "obj_ptr_proj",
            "obj_ptr_tpos_proj",
            "mask_downsample",
        ):
            if hasattr(self.sam_model, component_name):
                delattr(self.sam_model, component_name)
                self._removed_sam_components.append(component_name)
        for parameter_name in (
            "no_mem_embed",
            "no_mem_pos_enc",
            "no_obj_ptr",
            "no_obj_embed_spatial",
            "maskmem_tpos_enc",
        ):
            if hasattr(self.sam_model, parameter_name):
                delattr(self.sam_model, parameter_name)
                self._removed_sam_components.append(parameter_name)

        self.image_projection = nn.Sequential(
            nn.GroupNorm(1, image_feature_dim),
            nn.Conv2d(image_feature_dim, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=1),
        )
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(language_feature_dim),
            nn.Linear(language_feature_dim, 256),
        )

    @staticmethod
    def _bundled_source_root() -> Path:
        source_root = Path(__file__).resolve().parents[1] / "sam2"
        package_root = source_root / "sam2"
        if not (package_root / "configs").is_dir():
            raise FileNotFoundError(
                "Bundled SAM2 source was not found at "
                f"{source_root}. Initialize or restore vla/sam2."
            )
        return source_root

    @staticmethod
    def _install_isolated_sam2_namespace(source_root: Path) -> None:
        """Expose bundled SAM2 modules without executing its package initializer.

        Upstream SAM2 calls ``initialize_config_module`` from ``sam2/__init__.py``.
        That mutates Hydra's process-global state and conflicts with Habitat.  SAM2's
        model modules do not require that initializer, so register a package namespace
        pointing at the bundled source before Hydra resolves ``sam2.*`` targets from
        the YAML.  Python can then load the required submodules without running the
        upstream package body.
        """
        package_root = (source_root / "sam2").resolve()
        with SAM2MaskAuxiliaryHead._SAM2_NAMESPACE_LOCK:
            loaded_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "sam2" or name.startswith("sam2.")
            }
            existing_package = loaded_modules.get("sam2")
            if existing_package is not None:
                existing_paths = tuple(
                    Path(path).resolve()
                    for path in (getattr(existing_package, "__path__", ()) or ())
                )
                is_our_namespace = bool(
                    getattr(existing_package, "_trackvla_hydra_isolated", False)
                )
                foreign_submodules = tuple(
                    name
                    for name, module in loaded_modules.items()
                    if name != "sam2"
                    and not SAM2MaskAuxiliaryHead._is_bundled_sam2_module(
                        module, package_root
                    )
                )
                if (
                    is_our_namespace
                    and existing_paths == (package_root,)
                    and not foreign_submodules
                ):
                    return

                existing_location = getattr(existing_package, "__file__", None)
                if existing_location is None and existing_paths:
                    existing_location = ", ".join(map(str, existing_paths))
                details = (
                    f"; conflicting modules: {', '.join(foreign_submodules)}"
                    if foreign_submodules
                    else ""
                )
                raise RuntimeError(
                    "A `sam2` package was imported before TrackVLA installed its "
                    "Hydra-isolated bundled SAM2 namespace"
                    + (f" ({existing_location})" if existing_location else "")
                    + details
                    + ". Restart the process and do not import `sam2` before "
                    "constructing the TrackVLA model."
                )

            orphaned_submodules = tuple(
                name for name in loaded_modules if name.startswith("sam2.")
            )
            if orphaned_submodules:
                raise RuntimeError(
                    "SAM2 submodule(s) were imported before TrackVLA installed its "
                    "Hydra-isolated bundled SAM2 namespace: "
                    + ", ".join(orphaned_submodules)
                    + ". Restart the process and do not import `sam2` before "
                    "constructing the TrackVLA model."
                )

            package = types.ModuleType("sam2")
            package.__file__ = str(package_root / "__init__.py")
            package.__package__ = "sam2"
            package.__path__ = [str(package_root)]
            package.__spec__ = importlib.machinery.ModuleSpec(
                "sam2", loader=None, is_package=True
            )
            package.__spec__.submodule_search_locations = package.__path__
            package._trackvla_hydra_isolated = True
            sys.modules["sam2"] = package

    @staticmethod
    def _is_bundled_sam2_module(module: types.ModuleType, package_root: Path) -> bool:
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            return False
        try:
            Path(module_path).resolve().relative_to(package_root)
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _require_sam2_runtime_dependencies() -> None:
        """Fail clearly for dependencies Hydra would otherwise hide in its error."""
        try:
            from iopath.common.file_io import g_pathmgr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Bundled SAM2 imports `iopath.common.file_io` through "
                "`sam2.modeling.backbones.hieradet`. Install `iopath>=0.1.10` "
                "in the active environment, or install this repository's "
                "requirements.txt."
            ) from exc

    @staticmethod
    def _build_sam_model(
        *,
        config_file: str,
        checkpoint: Optional[str],
        source_root: Path,
    ) -> nn.Module:
        """Build an inference SAM2 model without Hydra's global compose API."""
        if checkpoint is None:
            raise ValueError("An enabled SAM2Config must include a checkpoint")
        SAM2MaskAuxiliaryHead._require_sam2_runtime_dependencies()
        try:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "Mask auxiliary training requires the bundled SAM2 runtime "
                "dependencies (hydra-core and iopath). Install requirements.txt."
            ) from exc

        config_path = SAM2MaskAuxiliaryHead._resolve_inference_config_path(
            config_file, source_root
        )
        checkpoint_path = SAM2MaskAuxiliaryHead._resolve_checkpoint_path(
            checkpoint, source_root
        )
        SAM2MaskAuxiliaryHead._install_isolated_sam2_namespace(source_root)

        # This is equivalent to SAM2's default ``apply_postprocessing=True``
        # overrides, but avoids ``hydra.compose`` and its GlobalHydra state.
        config = OmegaConf.load(config_path)
        if OmegaConf.select(config, "model") is None:
            raise ValueError(
                f"SAM2 config {config_path} must contain a top-level `model` section"
            )
        if OmegaConf.select(config, "model.sam_mask_decoder_extra_args") is None:
            OmegaConf.update(
                config,
                "model.sam_mask_decoder_extra_args",
                {},
                force_add=True,
            )
        for key, value in SAM2MaskAuxiliaryHead._DYNAMIC_MULTIMASK_DEFAULTS.items():
            OmegaConf.update(
                config,
                f"model.sam_mask_decoder_extra_args.{key}",
                value,
                force_add=True,
            )
        OmegaConf.resolve(config)
        model = instantiate(config.model, _recursive_=True)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"SAM2 config {config_path} instantiated {type(model).__name__}, "
                "not a torch.nn.Module"
            )
        SAM2MaskAuxiliaryHead._load_checkpoint_strict(model, checkpoint_path)
        model.to("cpu")
        model.eval()
        return model

    @classmethod
    def _resolve_inference_config_path(cls, config_file: str, source_root: Path) -> Path:
        package_root = (source_root / "sam2").resolve()
        inference_root = (package_root / "configs").resolve()
        requested_path = Path(config_file).expanduser()
        config_path = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (package_root / requested_path).resolve()
        )
        try:
            relative_path = config_path.relative_to(inference_root)
        except ValueError as exc:
            raise ValueError(
                "model.sam2_config.config_file must resolve under the bundled "
                f"SAM2 inference configs at {inference_root}, got {config_path}"
            ) from exc
        if (
            not config_path.is_file()
            or not relative_path.parts
            or relative_path.parts[0] not in cls._INFERENCE_CONFIG_DIRECTORIES
        ):
            raise ValueError(
                "model.sam2_config.config_file must name a bundled SAM2 inference "
                f"YAML under {inference_root / 'sam2'} or {inference_root / 'sam2.1'}, "
                f"got {config_path}"
            )
        return config_path

    @staticmethod
    def _resolve_checkpoint_path(checkpoint: str, source_root: Path) -> Path:
        requested_path = Path(checkpoint).expanduser()
        candidates = (
            (requested_path,)
            if requested_path.is_absolute()
            else (
                Path.cwd() / requested_path,
                source_root.parent.parent / requested_path,
                source_root / requested_path,
            )
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        searched_paths = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "SAM2 checkpoint was not found. Checked: " + searched_paths
        )

    @staticmethod
    def _load_checkpoint_strict(model: nn.Module, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
            raise RuntimeError(
                f"SAM2 checkpoint {checkpoint_path} must contain a `model` state dict"
            )
        state_dict = checkpoint["model"]
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"SAM2 checkpoint {checkpoint_path} has a non-mapping `model` entry"
            )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"SAM2 checkpoint {checkpoint_path} does not strictly match its YAML model"
            ) from exc

    @property
    def removed_sam_components(self) -> tuple[str, ...]:
        """Names discarded after checkpoint loading; useful for diagnostics/tests."""
        return tuple(self._removed_sam_components)

    @staticmethod
    def _configure_high_res_features(
        *,
        sam_model: nn.Module,
        decoder: nn.Module,
        requested: Optional[bool],
    ) -> bool:
        """Resolve the optional VLA high-resolution decoder path.

        The selected SAM2 YAML determines whether the checkpoint contains the
        ``conv_s0`` and ``conv_s1`` projections. An explicit ``False`` can
        safely disable a YAML-enabled path after strict checkpoint loading;
        enabling a YAML-disabled path is impossible because those projections
        were never constructed.
        """
        yaml_enables_high_res = bool(
            getattr(decoder, "use_high_res_features", False)
        )
        enabled = yaml_enables_high_res if requested is None else requested
        conv_s0 = getattr(decoder, "conv_s0", None)
        conv_s1 = getattr(decoder, "conv_s1", None)
        if yaml_enables_high_res and (
            not isinstance(conv_s0, nn.Module) or not isinstance(conv_s1, nn.Module)
        ):
            raise RuntimeError(
                "SAM2 mask decoder enables high-resolution features but does not "
                "expose conv_s0 and conv_s1 projections"
            )
        if enabled and not yaml_enables_high_res:
            raise ValueError(
                "model.sam2_config.use_high_res_features=True requires a SAM2 "
                "YAML with model.use_high_res_features_in_sam=true"
            )

        if not enabled and yaml_enables_high_res:
            # These projections are otherwise unused trainable parameters. Keep
            # their state for checkpoint/resume compatibility, but freeze them so
            # DDP does not expect gradients from the disabled branch.
            assert isinstance(conv_s0, nn.Module) and isinstance(conv_s1, nn.Module)
            conv_s0.requires_grad_(False)
            conv_s1.requires_grad_(False)

        decoder.use_high_res_features = enabled
        if hasattr(sam_model, "use_high_res_features_in_sam"):
            sam_model.use_high_res_features_in_sam = enabled
        return enabled

    @staticmethod
    def _decoder_high_res_features(
        image_embeddings: torch.Tensor,
        decoder: nn.Module,
    ) -> Optional[list[torch.Tensor]]:
        """Adapt VLA's single-scale visual map to SAM2's optional FPN inputs.

        SAM2.1's decoder expects projected level-0 and level-1 FPN features at
        four and two times the decoder grid size, respectively. The VLA does not
        retain SAM2's image encoder, so upsample its projected main feature map
        and reuse the checkpointed decoder projections instead.
        """
        if not bool(getattr(decoder, "use_high_res_features", False)):
            return None

        conv_s0 = getattr(decoder, "conv_s0", None)
        conv_s1 = getattr(decoder, "conv_s1", None)
        if not isinstance(conv_s0, nn.Module) or not isinstance(conv_s1, nn.Module):
            raise RuntimeError(
                "SAM2 mask decoder enables high-resolution features but does not "
                "expose conv_s0 and conv_s1 projections"
            )

        height, width = image_embeddings.shape[-2:]
        level_s0 = conv_s0(
            F.interpolate(
                image_embeddings,
                size=(height * 4, width * 4),
                mode="bilinear",
                align_corners=False,
            )
        )
        level_s1 = conv_s1(
            F.interpolate(
                image_embeddings,
                size=(height * 2, width * 2),
                mode="bilinear",
                align_corners=False,
            )
        )
        return [level_s0, level_s1]

    def decode(
        self,
        image_features: torch.Tensor,
        language_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode a mask without requiring a ground-truth mask label."""
        if image_features.ndim != 4:
            raise ValueError(f"image_features must have shape (B,C,H,W), got {tuple(image_features.shape)}")
        if language_feature.ndim != 2:
            raise ValueError(f"language_feature must have shape (B,D), got {tuple(language_feature.shape)}")
        if image_features.shape[0] != language_feature.shape[0]:
            raise ValueError("SAM2 auxiliary inputs must have the same batch dimension")

        image_features = F.interpolate(
            image_features,
            size=(self.feature_size, self.feature_size),
            mode="bilinear",
            align_corners=False,
        )
        decoder = self.sam_model.sam_mask_decoder
        decoder_dtype = next(decoder.parameters()).dtype
        image_projection_dtype = next(self.image_projection.parameters()).dtype
        prompt_projection_dtype = next(self.prompt_projection.parameters()).dtype
        image_embeddings = self.image_projection(
            image_features.to(dtype=image_projection_dtype)
        ).to(dtype=decoder_dtype)
        sparse_embeddings = self.prompt_projection(
            language_feature.to(dtype=prompt_projection_dtype)
        ).unsqueeze(1).to(dtype=decoder_dtype)
        batch_size = image_features.shape[0]
        dense_embeddings = self.empty_dense.expand(batch_size, -1, -1, -1).to(dtype=decoder_dtype)
        high_res_features = self._decoder_high_res_features(
            image_embeddings,
            decoder,
        )

        mask_logits, pred_iou, _, pred_obj_scores = decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_pe.to(dtype=decoder_dtype),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            high_res_features=high_res_features,
            repeat_image=False,
        )
        return {
            "mask_logits": mask_logits,
            "pred_masks": torch.sigmoid(mask_logits),
            "pred_iou": pred_iou,
            "pred_obj_scores": pred_obj_scores,
        }

    def compute_loss(
        self,
        decoded_masks: Mapping[str, torch.Tensor],
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute mask supervision for outputs from :meth:`decode`."""
        required_outputs = {"mask_logits", "pred_iou", "pred_obj_scores"}
        missing_outputs = required_outputs - set(decoded_masks)
        if missing_outputs:
            raise ValueError(
                "Decoded SAM2 masks are missing required output(s): "
                + ", ".join(sorted(missing_outputs))
            )
        if target_masks.ndim != 3:
            raise ValueError(f"target_masks must have shape (B,H,W), got {tuple(target_masks.shape)}")

        low_res_masks = decoded_masks["mask_logits"]
        pred_iou = decoded_masks["pred_iou"]
        pred_obj_scores = decoded_masks["pred_obj_scores"]
        if low_res_masks.ndim != 4:
            raise ValueError(
                "Decoded SAM2 mask_logits must have shape (B,C,H,W), got "
                f"{tuple(low_res_masks.shape)}"
            )
        if low_res_masks.shape[0] != target_masks.shape[0]:
            raise ValueError("SAM2 decoded masks and target masks must have the same batch dimension")

        targets = target_masks.to(
            device=low_res_masks.device, dtype=low_res_masks.dtype
        ).unsqueeze(1)
        mask_logits = F.interpolate(
            low_res_masks,
            size=tuple(targets.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, targets)
        dice_loss = _dice_loss(torch.sigmoid(mask_logits), targets)
        true_iou = _binary_iou(mask_logits, targets).detach().to(dtype=pred_iou.dtype)
        iou_loss = F.mse_loss(pred_iou, true_iou)
        object_target = (targets.flatten(1).amax(dim=1) > 0.5).to(dtype=pred_obj_scores.dtype)
        object_loss = F.binary_cross_entropy_with_logits(
            pred_obj_scores,
            object_target.view_as(pred_obj_scores),
        )
        loss = bce_loss + dice_loss + iou_loss + object_loss
        return {
            "mask_loss": loss,
            "mask_logits": mask_logits,
            "pred_masks": torch.sigmoid(mask_logits),
            "mask_iou": true_iou.mean(),
        }

    def forward(
        self,
        image_features: torch.Tensor,
        language_feature: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Preserve the labeled-training API on top of ``decode`` and ``compute_loss``."""
        decoded_masks = self.decode(image_features, language_feature)
        return {**decoded_masks, **self.compute_loss(decoded_masks, target_masks)}
