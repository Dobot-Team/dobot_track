#!/usr/bin/env python3
"""Render an eval visualization video for one Dobot training.json.

Loads a trained VLA checkpoint and, for every valid label in the video,
renders a frame showing:
  - left: current cam1 image
  - right: predicted future trajectory (red) vs ground truth (green)
    in the ego frame (up=x/forward, left=y/left)

Usage:
    python tools/vis_dobot_vla_eval_video.py \
        --checkpoint outputs/dobot_internvl2_5_vla_4epochs_bs2/checkpoints/last.ckpt \
        --json /mnt/cpfs/dobot_vla_local/post_sync_data/20260630/N-01-035-004-1782811378283823033/dobot_bag_0000_0/training.json \
        --output /tmp/dobot_vla_eval_video.mp4
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vla.data.dobot_vla import (
    _augment_scene_desc,
    find_nearest_frame_idx,
    interpolate_pose,
    transform_to_ego,
)
from vla.lightning import VLAPlModule
from vla.utils.checkpoint_config import (
    checkpoint_config_path,
    load_checkpoint_config,
)
from vla.utils.common import build_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training YAML override; defaults to config.yaml beside --checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .ckpt.")
    parser.add_argument("--json", type=str, required=True, help="Path to training.json.")
    parser.add_argument("--output", type=str, default="/tmp/dobot_vla_eval_video.mp4", help="Output MP4 path.")
    parser.add_argument("--cam_channel", type=str, default=None)
    parser.add_argument("--num_history_frames", type=int, default=None)
    parser.add_argument("--history_duration", type=float, default=None)
    parser.add_argument(
        "--n_future",
        type=int,
        default=None,
        help="Must match data.n_future; it cannot override the checkpoint architecture.",
    )
    parser.add_argument("--future_dt", type=float, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_image_tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10, help="Output video fps. Labels are ~10 Hz.")
    parser.add_argument("--max_frames", type=int, default=None, help="Render at most N frames.")
    parser.add_argument("--stride", type=int, default=1, help="Render one frame every N labels.")
    parser.add_argument("--traj_xlim", type=float, nargs=2, default=[-5.0, 5.0],
                        help="Fixed x-axis limits for trajectory plot (ego y, left-right).")
    parser.add_argument("--traj_ylim", type=float, nargs=2, default=[-1.0, 6.0],
                        help="Fixed y-axis limits for trajectory plot (ego x, backward-forward).")
    parser.add_argument("--no_autocast", action="store_true", help="Disable AMP autocast.")
    return parser.parse_args()


def load_json_data(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    return data


def denormalize(tensor: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = np.transpose(img, (1, 2, 0))
    return img


def draw_arrow(ax, x, y, yaw, length, color, linewidth=1.5):
    """Draw an arrow representing heading (yaw) in ego frame.

    Plot convention: ego_x -> upward, ego_y -> leftward.
    """
    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    ax.annotate(
        "",
        xy=(-(y + dy), x + dx),
        xytext=(-y, x),
        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth),
    )


def build_inputs(
    frames,
    labels,
    label_idx: int,
    transform,
    tokenizer,
    num_history_frames: int,
    history_duration: float,
    n_future: int,
    future_dt: float,
    cam_channel: str,
    prompt: str,
    num_image_tokens: int,
    max_length: int,
):
    """Build model inputs and ground-truth trajectory for a single label index."""
    frame_ts = np.array([f["timestamp"] * 1e-9 for f in frames], dtype=np.float64)
    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)
    history_offsets = np.linspace(-history_duration, 0.0, num_history_frames)
    future_offsets = np.arange(1, n_future + 1) * future_dt

    curr_label = labels[label_idx]
    curr_t = curr_label["timestamp"]
    curr_pose = np.array([
        curr_label["pose"]["x"],
        curr_label["pose"]["y"],
        curr_label["pose"]["yaw"],
    ], dtype=np.float32)

    # History images.
    history_ts = curr_t + history_offsets
    history_frame_idxs = [find_nearest_frame_idx(t, frame_ts) for t in history_ts]
    history_paths = [frames[idx][cam_channel] for idx in history_frame_idxs]
    images = [Image.open(p).convert("RGB") for p in history_paths]
    pixel_values = torch.stack([transform(img) for img in images], dim=0).unsqueeze(0)

    # Text prompt.
    image_token_str = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
    image_tokens = "\n".join([image_token_str] * num_history_frames)
    text = f"{image_tokens}\nUser: {prompt}\nAssistant:"
    tokenizer_out = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Ground truth trajectory in ego frame.
    future_ts = curr_t + future_offsets
    future_poses = np.array([interpolate_pose(t, labels) for t in future_ts], dtype=np.float32)
    traj_gt = transform_to_ego(curr_pose, future_poses)

    # Current image for visualization (latest history frame, before transform).
    current_img = images[-1]
    current_img_tensor = transform(current_img).numpy()
    current_img_vis = denormalize(current_img_tensor)

    return {
        "pixel_values": pixel_values,
        "input_ids": tokenizer_out["input_ids"],
        "attention_mask": tokenizer_out["attention_mask"],
        "traj_gt": traj_gt,
        "current_img_vis": current_img_vis,
        "current_img_path": history_paths[-1],
    }


def run_inference(module, inputs, dtype, use_autocast: bool):
    pixel_values = inputs["pixel_values"]  # (1, H, C, H_img, W_img)
    B, H, C, H_img, W_img = pixel_values.shape
    pixel_values = pixel_values.view(B * H, C, H_img, W_img).to(module.device)
    batch = {
        "pixel_values": pixel_values,
        "input_ids": inputs["input_ids"].to(module.device),
        "attention_mask": inputs["attention_mask"].to(module.device),
    }
    if use_autocast and module.device.type != "cpu":
        with torch.amp.autocast(module.device.type, dtype=dtype):
            with torch.no_grad():
                outputs = module(batch)
    else:
        with torch.no_grad():
            outputs = module(batch)
    pred = outputs["pred_traj"].float().cpu().numpy()[0]
    return pred


def render_frame(
    current_image: np.ndarray,
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    frame_label: str,
    args,
    cam_channel: str,
) -> np.ndarray:
    """Render a single visualization frame as RGB uint8 array."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: current image.
    ax_img = axes[0]
    ax_img.imshow(current_image)
    ax_img.set_title(f"Current frame ({cam_channel})\n{frame_label}")
    ax_img.axis("off")

    # Right: trajectory in ego frame.
    ax_traj = axes[1]
    ax_traj.set_aspect("equal", adjustable="box")

    # Plot convention: ego_x -> upward, ego_y -> leftward.
    pred_x = -pred_traj[:, 1]
    pred_y = pred_traj[:, 0]
    gt_x = -gt_traj[:, 1]
    gt_y = gt_traj[:, 0]

    ax_traj.plot(pred_x, pred_y, "r-o", markersize=3, linewidth=1.5, label="prediction")
    ax_traj.plot(gt_x, gt_y, "g-o", markersize=3, linewidth=1.5, label="ground truth")
    ax_traj.plot(0, 0, "k*", markersize=12, label="current ego")

    # Heading arrows every few points + last point.
    arrow_indices = list(range(0, len(pred_traj), 6)) + [len(pred_traj) - 1]
    for i in sorted(set(arrow_indices)):
        draw_arrow(ax_traj, pred_traj[i, 0], pred_traj[i, 1], pred_traj[i, 2], length=0.12, color="red")
        draw_arrow(ax_traj, gt_traj[i, 0], gt_traj[i, 1], gt_traj[i, 2], length=0.12, color="green")
    draw_arrow(ax_traj, 0.0, 0.0, 0.0, length=0.20, color="black", linewidth=2.0)

    ax_traj.set_xlim(args.traj_xlim)
    ax_traj.set_ylim(args.traj_ylim)
    ax_traj.set_xlabel("ego y (left)")
    ax_traj.set_ylabel("ego x (forward)")
    ax_traj.set_title("Future trajectory in ego frame\n(up=x, left=y)")
    ax_traj.legend(loc="upper left", fontsize=8)
    ax_traj.grid(True, linestyle="--", alpha=0.5)

    pred_final = pred_traj[-1]
    gt_final = gt_traj[-1]
    stats_text = (
        f"pred final: ({pred_final[0]:.2f}, {pred_final[1]:.2f}, {pred_final[2]:.2f})\n"
        f"gt final:   ({gt_final[0]:.2f}, {gt_final[1]:.2f}, {gt_final[2]:.2f})\n"
        f"ade: {np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=-1).mean():.3f}m"
    )
    ax_traj.text(
        0.98, 0.02, stats_text,
        transform=ax_traj.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return frame


def main():
    args = parse_args()

    config = load_checkpoint_config(args.checkpoint, args.config)
    if args.config is None:
        print(f"[vis_eval] Loaded config from {checkpoint_config_path(args.checkpoint)}")
    else:
        print(f"[vis_eval] Loaded config override from {args.config}")

    # Resolve overrides: CLI > config > defaults.
    cam_channel = args.cam_channel or config["data"].get("cam_channel", "cam1")
    num_history_frames = args.num_history_frames or config["data"].get("num_history_frames", 4)
    history_duration = args.history_duration if args.history_duration is not None else config["data"].get("history_duration", 1.0)
    n_future = int(config["data"]["n_future"])
    if args.n_future is not None and args.n_future != n_future:
        raise ValueError(
            f"--n_future ({args.n_future}) must match data.n_future ({n_future}) "
            "for the checkpoint being evaluated"
        )
    future_dt = args.future_dt if args.future_dt is not None else config["data"].get("future_dt", 0.25)
    input_size = args.input_size or config["data"].get("input_size", 448)
    max_length = args.max_length or config["data"].get("max_length", 2048)
    num_image_tokens = args.num_image_tokens or config["data"].get("num_image_tokens", 256)
    prompt = args.prompt or config["data"].get("prompt", "Predict the future trajectory of the ego vehicle.")

    dtype_str = config["model"].get("dtype", "bf16")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"[vis_eval] Loading checkpoint {args.checkpoint} ...")
    module = VLAPlModule.load_from_checkpoint(
        args.checkpoint,
        model_path=config["model"]["model_path"],
        model_type=config["model"].get("model_type", None),
        num_queries=config["model"]["num_queries"],
        n_future=n_future,
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
        dtype=dtype,
    )
    module.eval()
    module.freeze()
    print(f"[vis_eval] Model loaded on {module.device}")

    tokenizer = module.model.tokenizer
    transform = build_transform(input_size)

    use_scene_desc = config["data"].get("use_scene_desc", True)
    scene_desc_augment = config["data"].get("scene_desc_augment", True)
    scene_desc_seed = config["data"].get("seed", 42)

    print(f"[vis_eval] Loading {args.json} ...")
    data = load_json_data(args.json)
    frames = data.get("frames", [])
    labels = data.get("labels", [])
    scene_desc = data.get("bag_meta", {}).get("scene_desc", "")
    print(f"  frames: {len(frames)}, labels: {len(labels)}, scene_desc: {bool(scene_desc)}")
    if len(frames) == 0 or len(labels) == 0:
        print("Empty data, exiting.")
        return

    frame_ts = np.array([f["timestamp"] * 1e-9 for f in frames], dtype=np.float64)
    label_ts = np.array([l["timestamp"] for l in labels], dtype=np.float64)
    history_offsets = np.linspace(-history_duration, 0.0, num_history_frames)
    future_offsets = np.arange(1, n_future + 1) * future_dt

    valid_indices = []
    for i in range(0, len(labels), args.stride):
        curr_t = label_ts[i]
        if curr_t + history_offsets[0] < frame_ts[0]:
            continue
        if curr_t + future_offsets[-1] > label_ts[-1]:
            continue
        valid_indices.append(i)

    if args.max_frames is not None:
        valid_indices = valid_indices[:args.max_frames]

    print(f"[vis_eval] Rendering {len(valid_indices)} frames at {args.fps} fps -> "
          f"~{len(valid_indices)/args.fps:.1f}s video")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    use_autocast = not args.no_autocast

    # Render first frame to determine video size.
    idx = valid_indices[0]
    effective_prompt = prompt
    if use_scene_desc and scene_desc:
        if scene_desc_augment:
            token_str = f"{args.json}#{idx}#{scene_desc_seed}"
            rng = np.random.default_rng(abs(hash(token_str)) % (2**32))
            aug_desc = _augment_scene_desc(scene_desc, rng)
        else:
            aug_desc = scene_desc
        effective_prompt = f"{aug_desc} {prompt}"

    first_inputs = build_inputs(
        frames, labels, idx, transform, tokenizer,
        num_history_frames, history_duration, n_future, future_dt,
        cam_channel, effective_prompt, num_image_tokens, max_length,
    )
    first_pred = run_inference(module, first_inputs, dtype, use_autocast)
    first_frame = render_frame(
        first_inputs["current_img_vis"],
        first_pred,
        first_inputs["traj_gt"],
        f"frame {valid_indices[0]}",
        args,
        cam_channel,
    )
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    ade_list = []
    for idx in valid_indices[1:]:
        effective_prompt = prompt
        if use_scene_desc and scene_desc:
            if scene_desc_augment:
                token_str = f"{args.json}#{idx}#{scene_desc_seed}"
                rng = np.random.default_rng(abs(hash(token_str)) % (2**32))
                aug_desc = _augment_scene_desc(scene_desc, rng)
            else:
                aug_desc = scene_desc
            effective_prompt = f"{aug_desc} {prompt}"

        try:
            inputs = build_inputs(
                frames, labels, idx, transform, tokenizer,
                num_history_frames, history_duration, n_future, future_dt,
                cam_channel, effective_prompt, num_image_tokens, max_length,
            )
        except Exception as e:
            print(f"[vis_eval] Skipping frame {idx}: {e}")
            continue

        pred_traj = run_inference(module, inputs, dtype, use_autocast)
        gt_traj = inputs["traj_gt"]
        ade = np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=-1).mean()
        ade_list.append(ade)

        frame = render_frame(
            inputs["current_img_vis"],
            pred_traj,
            gt_traj,
            f"frame {idx}",
            args,
            cam_channel,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (len(ade_list) + 1) % 50 == 0:
            print(f"[vis_eval] Rendered {len(ade_list) + 1} / {len(valid_indices)} frames, "
                  f"mean ADE so far: {np.mean(ade_list):.3f}m")

    writer.release()
    print(f"[vis_eval] Video saved to {args.output}")
    if ade_list:
        print(f"[vis_eval] Mean ADE over rendered frames: {np.mean(ade_list):.4f}m")


if __name__ == "__main__":
    main()
