"""Shared paths, validated deployment scope and causal visual-feature cache."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--work", required=True, type=Path)
    return p


def model_parser(description):
    p = parser(description)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--sam", required=True, type=Path)
    p.add_argument(
        "--jetson", action="store_true", help="NVIDIA torch 2.5 SDPA compatibility"
    )
    return p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_config(cfg):
    """Keep the shipped Orin graph deliberately limited to the 1B recipe."""
    from vla.config import ModelExtensionsConfig, AuxiliaryTasksConfig

    m, d = cfg["model"], cfg["data"]
    ext = ModelExtensionsConfig.from_model_mapping(m)
    aux = AuxiliaryTasksConfig.from_value(m.get("auxiliary_tasks_config"))
    hc = ext.history_compressor
    if m.get("model_type") != "internvl3" or not hc.enabled or hc.mode != "q_identity":
        raise ValueError(
            "Expected InternVL3 + enabled q_identity (Joint alias accepted)"
        )
    if not hc.requires_history_boxes or hc.box_embedding:
        raise ValueError("Expected bbox target selection without box embedding")
    if not aux.grounding_enabled or aux.vqa_enabled:
        raise ValueError("Expected Box grounding enabled and VQA disabled")
    if m.get("future_frame_config", {}).get("enabled", False):
        raise ValueError("Future-frame prediction is not supported by this deployment")
    expected = {
        "num_history_frames": 8,
        "input_size": 448,
        "num_image_tokens": 256,
        "n_future": 8,
    }
    for key, value in expected.items():
        if d.get(key) != value:
            raise ValueError(f"data.{key} must be {value}, got {d.get(key)}")
    if int(d.get("current_frame_input_size") or 448) != 448:
        raise ValueError(
            "The shipped vision engine requires current-frame resolution 448"
        )
    if not 0 < int(d.get("max_length", 1200)) <= 32758:
        raise ValueError(
            "Prompt limit must leave all queries below Qwen's 32768-position base context"
        )
    if m.get("num_queries") != 8 or hc.num_output_tokens != 16:
        raise ValueError(
            "Expected eight Action queries and 16 tokens per history frame"
        )
    return ext, aux


def load(args):
    from scripts.eval_trackvla_habitat import load_vla_model, TrackVLAHabitatAgent

    if args.jetson:
        from transformers.integrations import sdpa_attention

        sdpa_attention._is_torch_greater_or_equal_than_2_5 = False
    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    cfg["model"]["model_path"] = str(args.base.resolve())
    cfg["model"]["sam2_config"]["checkpoint"] = str(args.sam.resolve())
    torch.set_num_threads(4)
    torch.manual_seed(100)
    model, kind, dtype = load_vla_model(
        cfg,
        args.checkpoint.resolve(),
        str(args.base.resolve()),
        None,
        None,
        torch.device("cuda"),
        False,
    )
    if model.llm_hidden_size != 896:
        raise ValueError(
            "The TRT-LLM graph supports InternVL3-1B / Qwen2 hidden size 896 only"
        )
    model.requires_grad_(False).eval()
    agent = TrackVLAHabitatAgent(
        model,
        kind,
        cfg["data"],
        torch.device("cuda"),
        dtype,
        0.1,
        0,
        True,
        "Track the person in front of you.",
    )
    receipt = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "config_sha256": sha256(args.config),
        "query_order": (
            ["mask", "box", "action_x8"]
            if model.mask_query_token is not None
            else ["box", "action_x8"]
        ),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    receipt_path = root / "provenance.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if any(
            old.get(k) != receipt[k]
            for k in ("checkpoint_sha256", "config_sha256", "query_order", "git_commit")
        ):
            raise ValueError(
                "Work directory belongs to a different checkpoint/config/query order/source revision; use a new --work"
            )
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    (root / "training_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root, cfg, model, agent


class StreamingVision:
    """Encode one new RGB frame; retain only features referenced by the agent."""

    def __init__(self, model, agent, encoder=None):
        self.agent = agent
        self.encoder = encoder or model.internvl.get_image_features
        self.preprocess = agent._internvl_pixel_values
        self.features = {}
        self.window = None
        self.new_frames = 0
        self.total_source = 0
        agent._internvl_pixel_values = self.prepare
        model.internvl.get_image_features = self.get

    def prepare(self, frames):
        source = list(self.agent.frames)
        alive = {id(x) for x in source}
        self.features = {k: v for k, v in self.features.items() if k in alive}
        keys = [id(source[i]) for i in self.agent._history_indices()]
        for frame, key in zip(frames, keys):
            if key not in self.features:
                pixels = self.preprocess([frame])
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    self.features[key] = self.encoder(pixel_values=pixels)
                self.new_frames += 1
                self.total_source += 1
        self.window = torch.cat([self.features[key] for key in keys], 0)
        return torch.empty(0, device="cuda", dtype=torch.bfloat16)

    def get(self, **kwargs):
        return self.window

    def clear(self):
        self.features.clear()
        self.window = None
