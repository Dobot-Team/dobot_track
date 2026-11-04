#!/usr/bin/env python3
"""Validate a real dataset converted into TrackVLA normalized target space."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from resample_real_trackvla import iter_json_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = 0
    invalid_history = 0
    invalid_trajectory = 0
    invalid_actions = 0
    out_of_range = 0
    nonfinite = 0
    max_abs_action = [0.0, 0.0, 0.0]
    clipped_components = [0, 0, 0]
    history_padding = 0

    for record in iter_json_array(args.dataset):
        count += 1
        history = record.get("history_frames") or []
        history_valid = record.get("history_frame_valid") or []
        trajectory = record.get("trajectory") or []
        actions = record.get("actions_normalized") or []

        if len(history) != 8 or len(history_valid) != 8:
            invalid_history += 1
        history_padding += sum(not bool(value) for value in history_valid)
        if len(trajectory) != 8 or any(len(point) != 3 for point in trajectory):
            invalid_trajectory += 1
        if len(actions) != len(trajectory) or any(len(action) != 3 for action in actions):
            invalid_actions += 1
            continue

        for action in actions:
            for axis, value in enumerate(action):
                value = float(value)
                if not math.isfinite(value):
                    nonfinite += 1
                    continue
                max_abs_action[axis] = max(max_abs_action[axis], abs(value))
                if abs(value) > 1.0 + 1e-6:
                    out_of_range += 1
                if abs(value) >= 1.0 - 1e-7:
                    clipped_components[axis] += 1

        if count % 100000 == 0:
            print(f"[validate] records={count}", flush=True)

    summary = {
        "dataset": str(args.dataset),
        "records": count,
        "invalid_history_records": invalid_history,
        "invalid_trajectory_records": invalid_trajectory,
        "invalid_action_records": invalid_actions,
        "nonfinite_components": nonfinite,
        "out_of_range_components": out_of_range,
        "max_abs_normalized_action": max_abs_action,
        "clipped_components": {
            "forward": clipped_components[0],
            "lateral": clipped_components[1],
            "angular": clipped_components[2],
        },
        "history_padding_slots": history_padding,
        "valid": all(
            value == 0
            for value in (
                invalid_history,
                invalid_trajectory,
                invalid_actions,
                nonfinite,
                out_of_range,
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
