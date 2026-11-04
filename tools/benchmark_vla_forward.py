#!/usr/bin/env python3
"""Benchmark a single forward+backward pass of the InternVL VLA model."""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from vla.models import InternVLVLA


def main():
    model_path = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"
    batch_size = 2
    num_history = 4
    n_future = 8
    max_length = 1200
    num_image_tokens = 256
    device = torch.device("cuda")

    print("Loading model...")
    t0 = time.time()
    model = InternVLVLA(
        model_path=model_path,
        num_queries=n_future,
        freeze_vision=True,
        freeze_llm=False,
        freeze_mlp1=True,
        action_head_dims=(512, 3),
        dtype=torch.bfloat16,
        n_future=n_future,
    )
    model.to(device)
    model.train()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Build dummy inputs mimicking the real training batch.
    pixel_values = torch.randn(batch_size * num_history, 3, 448, 448, dtype=torch.bfloat16, device=device)

    image_token_str = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
    image_tokens = "\n".join([image_token_str] * num_history)
    text = f"{image_tokens}\nUser: Predict the future trajectory of the ego vehicle.\nAssistant:"
    tokenizer = model.tokenizer
    tok = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = tok["input_ids"].expand(batch_size, -1).to(device)
    attention_mask = tok["attention_mask"].expand(batch_size, -1).to(device)
    trajectories = torch.randn(batch_size, n_future, 3, device=device)

    # Warmup
    print("Warmup forward...")
    with torch.no_grad():
        _ = model(pixel_values, input_ids, attention_mask)

    # Forward only
    print("Timing forward...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        out = model(pixel_values, input_ids, attention_mask)
        torch.cuda.synchronize()
    fwd_time = (time.time() - t0) / 3
    print(f"Forward time: {fwd_time:.2f}s")

    # Forward + backward
    print("Timing forward+backward...")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(pixel_values, input_ids, attention_mask, trajectories)
        out["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize()
    step_time = (time.time() - t0) / 3
    print(f"Forward+backward time: {step_time:.2f}s")


if __name__ == "__main__":
    main()
