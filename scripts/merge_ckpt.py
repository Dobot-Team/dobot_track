#!/usr/bin/env python3
"""Merge a metadata-preserving VLA inference checkpoint.

Example:
    python scripts/merge_ckpt.py \
        --checkpoint outputs/trackvla/checkpoints/last.ckpt/ \
        --output outputs/trackvla/checkpoints/last_inference.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla.lightning.checkpoint import merge_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate a VLA checkpoint while preserving Lightning "
            "hyperparameters and extension metadata."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning .ckpt file or DeepSpeed checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .ckpt path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing destination file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_path = Path(args.output).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    merged_path = merge_checkpoint(
        checkpoint_path,
        output_path,
    )
    print(
        "[checkpoint-merge] Wrote metadata-preserving inference checkpoint: "
        f"{merged_path}"
    )


if __name__ == "__main__":
    main()
