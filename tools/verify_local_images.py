#!/usr/bin/env python3
"""Scan local Dobot mirror and report corrupted/truncated images."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image


def _check_video(json_path):
    bad = []
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return (json_path, [("json", str(e))])
    for frame in data.get("frames", []):
        p = frame.get("cam1")
        if not p or not os.path.exists(p):
            continue
        try:
            with Image.open(p) as img:
                img.load()
        except Exception as e:
            bad.append((p, str(e)))
    return (json_path, bad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_list", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.json_list) as f:
        json_paths = [l.strip() for l in f if l.strip()]

    total_bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_check_video, p): p for p in json_paths}
        for fut in as_completed(futures):
            json_path, bad = fut.result()
            if bad:
                total_bad += len(bad)
                print(f"[BAD] {json_path}: {len(bad)} bad images")
                for p, e in bad[:5]:
                    print(f"      {p}: {e}")
    print(f"\nTotal bad images: {total_bad}")


if __name__ == "__main__":
    main()
