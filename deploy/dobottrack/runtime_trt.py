"""TensorRT policy compatible with TrackVLAHabitatAgent's causal Box cache."""

import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import tensorrt as trt


def is_host_input(name, location):
    return name.startswith("host_") or location == trt.TensorLocation.HOST


class Runner:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Cannot deserialize engine {path}; rebuild on target TensorRT/GPU"
            )
        self.context = self.engine.create_execution_context()
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.inputs = [
            n
            for n in self.names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.buffers = {}
        self.pending = []

    def __call__(self, inputs):
        # Retain temporary casts/host buffers until the prior asynchronous enqueue has finished.
        self.pending = [
            (event, values) for event, values in self.pending if not event.query()
        ]
        keep = []
        for name in self.inputs:
            dtype = torch.from_numpy(
                np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            location = self.engine.get_tensor_location(name)
            host = name.startswith("host_") or location == trt.TensorLocation.HOST
            value = (
                inputs[name]
                .to(dtype=dtype, device="cpu" if host else "cuda")
                .contiguous()
            )
            if host and not value.is_pinned():
                value = value.pin_memory()
            keep.append(value)
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f"Input outside engine profile: {name} {value.shape}")
            self.context.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for name in self.names:
            if name in self.inputs:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            if any(n < 0 for n in shape):
                raise RuntimeError(f"Unresolved output shape: {name} {shape}")
            dtype = torch.from_numpy(
                np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            key = name, shape, dtype
            if key not in self.buffers:
                self.buffers[key] = torch.empty(shape, dtype=dtype, device="cuda")
            outputs[name] = self.buffers[key]
            self.context.set_tensor_address(name, outputs[name].data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        event = torch.cuda.Event()
        event.record()
        self.pending.append((event, keep))
        return outputs


def metadata(length, spec):
    def host(x, dtype=torch.int32):
        return torch.tensor(x, dtype=dtype, pin_memory=True)

    max_seq = spec["max_sequence_length"]
    values = {
        "input_ids": torch.zeros(1, length, dtype=torch.int32, device="cuda"),
        "position_ids": torch.arange(length, device="cuda", dtype=torch.int32)[None],
        "last_token_ids": torch.tensor([length], device="cuda", dtype=torch.int32),
        "sequence_length": torch.tensor([length], device="cuda", dtype=torch.int32),
        "context_lengths": torch.tensor([length], device="cuda", dtype=torch.int32),
        "host_context_lengths": host([length]),
        "host_request_types": host([0]),
        "host_past_key_value_lengths": host([0]),
        "host_max_attention_window_sizes": host([max_seq] * spec["llm_layers"]),
        "host_sink_token_length": host([0]),
        "host_runtime_perf_knobs": host([-1] * 16, torch.int64),
        "cache_indirection": torch.zeros(
            1, 1, max_seq, device="cuda", dtype=torch.int32
        ),
    }
    # The graph disables KV reuse; allocate these only if TRT-LLM retained them as engine inputs.
    return values


class Policy(nn.Module):
    def __init__(
        self,
        root,
        base,
        config,
        qformer=None,
        vision=None,
        decode_mask=False,
        recipe="optimized",
    ):
        super().__init__()
        import tensorrt_llm  # Register native plugins before engine deserialization.

        plugin = os.environ.get("DOBOTTRACK_LLM_PLUGIN")
        manifest_path = Path(root) / "engine_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        expected_plugin = manifest.get("plugin_sha256")
        if expected_plugin and not plugin:
            plugin = str(Path(root) / "lib/libtrackvla_llm_norm_quant_plugin.so")
        if expected_plugin:
            from deploy.dobottrack.common import sha256

            if not plugin or sha256(plugin) != expected_plugin:
                raise ValueError(
                    "Custom fusion plugin differs from build manifest; set DOBOTTRACK_LLM_PLUGIN to the matching library"
                )
        if plugin:
            ctypes.CDLL(plugin)
        from vla.config import ModelExtensionsConfig, AuxiliaryTasksConfig
        from transformers import AutoTokenizer
        from deploy.dobottrack.common import validate_config

        validate_config(config)
        root = Path(root)
        self.spec = json.loads((root / "deployment.json").read_text())
        provenance = json.loads((root / "provenance.json").read_text())
        expected = (
            ["mask", "box", "action_x8"]
            if self.spec["mask_query"]
            else ["box", "action_x8"]
        )
        if (
            provenance["query_order"] != expected
            or self.spec["provenance"] != provenance
        ):
            raise ValueError(
                "Export provenance/query order mismatch; re-export all artifacts"
            )
        self.model_extensions = ModelExtensionsConfig.from_model_mapping(
            config["model"]
        )
        self.auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            config["model"]["auxiliary_tasks_config"]
        )
        self.sam2_config = self.model_extensions.sam2
        self.history_compressor_config = self.model_extensions.history_compressor
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.dtype = torch.bfloat16
        self.img_context_token_id = int(
            json.loads((Path(base) / "config.json").read_text())["image_token_id"]
        )
        self.recipe = recipe
        qformer = qformer or manifest["qformer"]
        vision = vision or manifest["vision"]
        if recipe == "optimized" and (
            qformer != "qformer_compact_fp" or not manifest.get("plugin_sha256")
        ):
            raise ValueError(
                "Optimized execution needs compact QFormer and fused LLM engines; rebuild with --recipe optimized"
            )
        self.measure_stages = False
        self.timings = {k: [] for k in ["vision", "qformer", "llm", "heads", "mask"]}
        self.vision = Runner(root / f"engines/{vision}.engine")
        if recipe == "optimized":
            from deploy.dobottrack.compact_runtime import CompactFastRunner

            self.qformer = CompactFastRunner(
                root / "engines/qformer_compact_fp.engine", Runner
            )
        else:
            self.qformer = Runner(root / f"engines/{qformer}.engine")
        self.llm = Runner(root / "engines/llm/rank0.engine")
        self.heads = Runner(root / "engines/heads.engine")
        self.mask = (
            Runner(root / "engines/mask.engine")
            if decode_mask and self.spec["mask_query"]
            else None
        )
        self.internvl = SimpleNamespace(get_image_features=self.encode)
        self.templates = {}
        with np.load(root / "templates.npz") as data:
            for item in json.loads((root / "templates.json").read_text()):
                i = item["index"]
                key = tuple(data[f"ids_{i}"][0].tolist())
                self.templates[key] = (
                    torch.from_numpy(data[f"template_{i}"]).cuda(),
                    torch.from_numpy(data[f"slots_{i}"]).cuda(),
                )
        self.meta = {}
        if recipe == "optimized":
            from deploy.dobottrack.optimized import (
                ShapeCache,
                BoundRunner,
                graph_evidence,
            )

            self.vision = ShapeCache(self.vision, True)
            self.llm = ShapeCache(self.llm, True)
            self.heads = ShapeCache(self.heads, True)
            for k, (runner, hi, hp, feed) in list(self.qformer.buckets.items()):
                self.qformer.buckets[k] = (
                    BoundRunner(runner, feed, True),
                    hi,
                    hp,
                    feed,
                )
            for template, slots in self.templates.values():
                n = template.shape[1]
                self.meta[n] = self.make_metadata(n)
                hidden = self.llm(dict(self.meta[n], inputs_embeds=template))[
                    "hidden_states"
                ]
                self.heads({"queries": hidden[:, -self.spec["query_count"] :]})
            self.vision({"pixel_values": torch.zeros((1, 3, 448, 448), device="cuda")})
            torch.cuda.synchronize()
            failed = [row for row in graph_evidence(self) if not row["captured"]]
            if failed:
                raise RuntimeError(f"CUDA Graph capture failed: {failed}")

    def make_metadata(self, n):
        values = metadata(n, self.spec)
        runner = getattr(self.llm, "base", self.llm)
        for name in runner.inputs:
            if name.startswith("past_key_value_"):
                values[name] = torch.zeros(
                    1,
                    2,
                    self.spec["kv_heads"],
                    self.spec["max_sequence_length"],
                    self.spec["head_size"],
                    dtype=torch.float16,
                    device="cuda",
                )
        return values

    def timed(self, name, runner, inputs):
        import time

        if self.measure_stages:
            torch.cuda.synchronize()
            start = time.perf_counter()
        result = runner(inputs)
        if self.measure_stages:
            torch.cuda.synchronize()
            self.timings[name].append((time.perf_counter() - start) * 1000)
        return result

    def encode(self, *, pixel_values):
        return self.timed("vision", self.vision, {"pixel_values": pixel_values})[
            "image_features"
        ].clone()

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        history_boxes,
        history_box_valid,
        **kwargs,
    ):
        features = self.internvl.get_image_features(pixel_values=pixel_values)
        compressed = self.timed(
            "qformer",
            self.qformer,
            {
                "features": features[:-1],
                "boxes": history_boxes[0],
                "valid": history_box_valid[0],
            },
        )["compressed_tokens"]
        key = tuple(input_ids[0].tolist())
        if key not in self.templates:
            raise ValueError(
                "Instruction was not exported; add it to --instructions and re-export templates"
            )
        template, slots = self.templates[key]
        embeds = template.clone()
        embeds[0, slots] = torch.cat((compressed.reshape(-1, 896), features[-1]), 0).to(
            embeds
        )
        n = embeds.shape[1]
        if n > self.spec["max_sequence_length"]:
            raise ValueError("Prompt exceeds exported maximum sequence length")
        if n not in self.meta:
            self.meta[n] = self.make_metadata(n)
        feed = dict(self.meta[n], inputs_embeds=embeds)
        hidden = self.timed("llm", self.llm, feed)["hidden_states"][
            :, -self.spec["query_count"] :
        ]
        result = self.timed("heads", self.heads, {"queries": hidden})
        logits = (
            self.mask({"features": features[-1:], "query": hidden[:, 0]})["mask_logits"]
            if self.mask
            else None
        )
        if self.recipe == "optimized":
            from deploy.dobottrack.optimized import packed_cpu_outputs

            result = packed_cpu_outputs(result)
        return dict(
            pred_traj=result["trajectory"],
            grounding_bbox=result["grounding_bbox"],
            grounding_visibility_logits=result["grounding_visibility_logits"],
            mask_logits=logits,
        )
