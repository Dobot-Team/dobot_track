#!/usr/bin/env python3
"""Create a TRT-LLM SmoothQuant W8A8 checkpoint using TrackVLA embeddings."""

from __future__ import annotations

import argparse
import functools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.qwen.config import QWenConfig
from tensorrt_llm.models.qwen.convert import (
    load_weights_from_hf_model,
    smooth_qwen2_model,
)
from tensorrt_llm.quantization.mode import QuantAlgo


@torch.inference_mode()
def capture_trackvla_activation_ranges(
    model: torch.nn.Module, calibration_path: Path, samples: int
) -> defaultdict:
    """Collect SmoothQuant ranges from visual/text/query planner embeddings."""
    data = np.load(calibration_path)
    embeddings = data["inputs_embeds"]
    masks = data["attention_mask"]
    positions = data["position_ids"]
    if samples < 1 or samples > len(embeddings):
        raise ValueError(f"--samples must be in [1, {len(embeddings)}].")

    ranges = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def record(name: str, tensor: torch.Tensor, key: str) -> None:
        values = tensor.reshape(-1, tensor.shape[-1]).abs().detach()
        current = values.max(dim=0).values.float()
        ranges[name][key] = (
            current
            if ranges[name][key] is None
            else torch.maximum(ranges[name][key], current)
        )

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        name: str,
    ) -> None:
        record(name, inputs[0], "x")
        record(name, output, "y")
        if ranges[name]["w"] is None:
            ranges[name]["w"] = module.weight.abs().clamp_min(1e-8).max(dim=1).values

    hooks = [
        module.register_forward_hook(functools.partial(hook, name=name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    try:
        for index in range(samples):
            valid = masks[index].astype(bool)
            length = int(valid.sum())
            model(
                inputs_embeds=torch.from_numpy(embeddings[index : index + 1, valid])
                .cuda()
                .half(),
                attention_mask=torch.ones((1, length), device="cuda", dtype=torch.long),
                position_ids=torch.arange(
                    length, device="cuda", dtype=torch.long
                ).unsqueeze(0),
                use_cache=False,
            )
            if (index + 1) % 16 == 0 or index + 1 == samples:
                print(f"[Calibration] {index + 1}/{samples}")
    finally:
        for registered_hook in hooks:
            registered_hook.remove()
    return ranges


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
        smoothquant_val=args.smoothing_strength,
    )
    config = QWenConfig.from_hugging_face(
        args.model_dir,
        dtype="float16",
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        quant_config=quant_config,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
        )
        .cuda()
        .eval()
    )
    activation_ranges = capture_trackvla_activation_ranges(
        model, Path(args.calibration), args.samples
    )
    qkv_parameters: dict = {}
    smoothers: dict = {}
    smooth_qwen2_model(
        model,
        activation_ranges,
        quant_config.smoothquant_val,
        qkv_parameters,
        smoothers,
    )
    weights = load_weights_from_hf_model(
        model,
        config=config,
        act_range=activation_ranges,
        qkv_para=qkv_parameters,
        smoother=smoothers,
    )
    config.to_json_file(output_dir / "config.json")
    safetensors.torch.save_file(weights, output_dir / "rank0.safetensors")
    print(f"[Output] Saved dynamic per-token W8A8 TRT-LLM checkpoint to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--smoothing-strength", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
