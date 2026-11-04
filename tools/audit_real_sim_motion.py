#!/usr/bin/env python3
"""Audit real Dobot and TrackVLA simulator motion on comparable units.

Real motion is recovered from consecutive timestamped poses and rotated into
the robot frame. Simulator motion reports stored normalized actions and a
label-clock equivalent metric velocity::

    u = clip(action, -1, 1)
    delta_pose = u * speed_coefficients * action_integration_dt
    label_clock_equivalent_velocity = delta_pose / label_dt

This last quantity is not the simulator's instantaneous physical velocity.
The physical command is ``u * speed_coefficients`` and one Habitat env step
moves for ``1 / ctrl_freq`` seconds. Dividing that one-step displacement by
the dataset builder's synthetic 0.1-second label interval is useful only when
comparing the spatial target encoded by one label step.

Only the first action of each TrackVLA training sample is used.  Future action
windows overlap between adjacent samples, so pooling every future action would
count the same controller step repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


FIELDS = ("forward", "lateral", "linear_norm", "angular", "abs_angular", "dt")
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("/home/jovyan/work/post_sync_res/pipeline_v2"),
    )
    parser.add_argument(
        "--real-dataset",
        type=Path,
        default=None,
        help=(
            "Stream an already converted real dataset.json instead of "
            "enumerating raw training.json files. Its metric trajectory is "
            "assumed to use --record-dt spacing."
        ),
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("/home/jovyan/work/dataset/sample"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--real-max-files", type=int, default=0)
    parser.add_argument(
        "--real-file",
        type=Path,
        action="append",
        default=[],
        help="Audit explicit training.json file(s), mainly for smoke tests.",
    )
    parser.add_argument("--sim-max-records", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-real-dt", type=float, default=0.02)
    parser.add_argument("--max-real-dt", type=float, default=0.30)
    parser.add_argument("--record-dt", type=float, default=0.10)
    parser.add_argument("--integration-dt", type=float, default=0.025)
    return parser.parse_args()


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def iter_json_array(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                block = handle.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected top-level JSON array: {path}")
                position += 1
                started = True
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"Unterminated JSON array: {path}")
                continue
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                block = handle.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True
                continue
            position = end
            if isinstance(item, dict):
                yield item


def new_values() -> Dict[str, array]:
    return {field: array("f") for field in FIELDS}


def append_motion(values: MutableMapping[str, array], vx: float, vy: float, omega: float, dt: float) -> None:
    if not all(math.isfinite(v) for v in (vx, vy, omega, dt)):
        return
    values["forward"].append(vx)
    values["lateral"].append(vy)
    values["linear_norm"].append(math.hypot(vx, vy))
    values["angular"].append(omega)
    values["abs_angular"].append(abs(omega))
    values["dt"].append(dt)


def summarize(values: Mapping[str, array]) -> dict:
    result: Dict[str, Any] = {"count": len(values["forward"]), "metrics": {}}
    for field in FIELDS:
        data = np.frombuffer(values[field], dtype=np.float32)
        if not len(data):
            result["metrics"][field] = None
            continue
        q = np.quantile(data, QUANTILES)
        result["metrics"][field] = {
            "mean": float(np.mean(data, dtype=np.float64)),
            "std": float(np.std(data, dtype=np.float64)),
            "min": float(q[0]),
            "p50": float(q[1]),
            "p90": float(q[2]),
            "p95": float(q[3]),
            "p99": float(q[4]),
            "p99_9": float(q[5]),
            "max": float(q[6]),
        }
    if result["count"]:
        forward = np.frombuffer(values["forward"], dtype=np.float32)
        linear = np.frombuffer(values["linear_norm"], dtype=np.float32)
        angular = np.frombuffer(values["abs_angular"], dtype=np.float32)
        result["fractions"] = {
            "stationary_linear_lt_0.05": float(np.mean(linear < 0.05)),
            "reverse_forward_lt_0": float(np.mean(forward < 0.0)),
            "turning_abs_angular_ge_0.5": float(np.mean(angular >= 0.5)),
        }
    return result


def timestamp_seconds(label: Mapping[str, Any]) -> float:
    value = float(label["timestamp"])
    return value * 1e-9 if value > 1e12 else value


def audit_real(args: argparse.Namespace) -> tuple[dict, Dict[str, array]]:
    # The raw tree also contains millions of RGB/mask files.  ``rglob`` would
    # descend into all of them on network storage; training.json always lives
    # at <date>/<session>/<bag>/training.json, so use the bounded layout.
    paths = list(args.real_file) if args.real_file else sorted(
        args.real_root.glob("*/*/*/training.json")
    )
    if args.real_max_files:
        paths = paths[: args.real_max_files]
    values = new_values()
    annotated = new_values()
    invalid_dt = 0
    malformed = 0
    episode_counts = []
    for file_index, path in enumerate(paths, 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = payload.get("labels", [])
            episode_valid = 0
            for first, second in zip(labels, labels[1:]):
                dt = timestamp_seconds(second) - timestamp_seconds(first)
                if not (args.min_real_dt <= dt <= args.max_real_dt):
                    invalid_dt += 1
                    continue
                p0 = first["pose"]
                p1 = second["pose"]
                dx = (float(p1["x"]) - float(p0["x"])) / dt
                dy = (float(p1["y"]) - float(p0["y"])) / dt
                yaw = float(p0["yaw"])
                vx = math.cos(yaw) * dx + math.sin(yaw) * dy
                vy = -math.sin(yaw) * dx + math.cos(yaw) * dy
                omega = wrap_angle(float(p1["yaw"]) - yaw) / dt
                append_motion(values, vx, vy, omega, dt)
                velocity = first.get("velocity") or {}
                if all(key in velocity for key in ("vx", "vy", "omega")):
                    append_motion(
                        annotated,
                        float(velocity["vx"]),
                        float(velocity["vy"]),
                        float(velocity["omega"]),
                        dt,
                    )
                episode_valid += 1
            episode_counts.append(episode_valid)
        except Exception as exc:
            malformed += 1
            print(f"[real] skip {path}: {exc}", flush=True)
        if file_index % args.progress_every == 0 or file_index == len(paths):
            print(f"[real] {file_index}/{len(paths)} files, motions={len(values['forward'])}", flush=True)
    report = {
        "source": str(args.real_root),
        "files": len(paths),
        "malformed_files": malformed,
        "invalid_dt_pairs": invalid_dt,
        "pose_derived_robot_frame": summarize(values),
        "provided_velocity_as_stored": summarize(annotated),
        "valid_pairs_per_file": {
            "min": min(episode_counts, default=0),
            "median": float(np.median(episode_counts)) if episode_counts else 0,
            "max": max(episode_counts, default=0),
        },
    }
    return report, values


def audit_real_dataset(args: argparse.Namespace) -> tuple[dict, Dict[str, array]]:
    """Audit first-step metric trajectories from converted real data.

    Adjacent manifest samples overlap, so using only the first future point
    counts each current motion once, just like the simulation audit.
    """

    values = new_values()
    missing = 0
    count = 0
    for record in iter_json_array(args.real_dataset):
        trajectory = record.get("trajectory") or []
        if not trajectory or len(trajectory[0]) < 3:
            missing += 1
            continue
        point = [float(value) for value in trajectory[0][:3]]
        append_motion(
            values,
            point[0] / args.record_dt,
            point[1] / args.record_dt,
            point[2] / args.record_dt,
            args.record_dt,
        )
        count += 1
        if count % 100000 == 0:
            print(f"[real-converted] records={count}", flush=True)
    report = {
        "source": str(args.real_dataset),
        "files": 1,
        "missing_trajectory_records": missing,
        "pose_derived_robot_frame": summarize(values),
        "provided_velocity_as_stored": None,
        "note": (
            "Velocity reconstructed from the first metric trajectory point "
            "divided by record_dt; raw pose/timestamp audit is preferred when "
            "the filesystem can enumerate source episodes."
        ),
    }
    return report, values


SIM_SOURCES = {
    "at_train_60": (15.0, 5.0, 6.28),
    "dt_train_0.65": (15.0, 10.0, 6.28),
    "dt_train_60": (15.0, 10.0, 6.28),
    "stt_train_70": (15.0, 10.0, 6.28),
}


def audit_sim(args: argparse.Namespace) -> tuple[dict, Dict[str, array]]:
    combined_raw = new_values()
    combined_effective = new_values()
    result: Dict[str, Any] = {}
    for source, coefficients in SIM_SOURCES.items():
        path = args.sim_root / source / "dataset.json"
        raw = new_values()
        effective = new_values()
        count = 0
        for record in iter_json_array(path):
            actions = record.get("actions") or []
            if not actions or len(actions[0]) < 3:
                continue
            action = [float(v) for v in actions[0][:3]]
            append_motion(raw, action[0], action[1], action[2], args.record_dt)
            append_motion(
                combined_raw, action[0], action[1], action[2], args.record_dt
            )
            clipped = [max(-1.0, min(1.0, value)) for value in action]
            metric = [
                clipped[index] * coefficients[index] * args.integration_dt / args.record_dt
                for index in range(3)
            ]
            append_motion(effective, metric[0], metric[1], metric[2], args.record_dt)
            append_motion(combined_effective, metric[0], metric[1], metric[2], args.record_dt)
            count += 1
            if args.sim_max_records and count >= args.sim_max_records:
                break
            if count % 100000 == 0:
                print(f"[sim:{source}] records={count}", flush=True)
        result[source] = {
            "path": str(path),
            "speed_coefficients": list(coefficients),
            "normalized_action": summarize(raw),
            "effective_metric_velocity": summarize(effective),
        }
        print(f"[sim:{source}] complete records={count}", flush=True)
    return {
        "sources": result,
        "combined_normalized_action": summarize(combined_raw),
        "combined_label_clock_equivalent_metric_velocity": summarize(
            combined_effective
        ),
    }, combined_raw


def normalization_scale_candidates(
    real: Mapping[str, array], sim: Mapping[str, array]
) -> dict:
    """Return real m/s (or rad/s) represented by normalized command 1.0.

    The strict-control-variable conversion leaves simulation targets untouched.
    For each axis, scale = real absolute-velocity quantile / simulation
    absolute-action quantile. Applying ``real_velocity / scale`` therefore
    matches the chosen marginal action quantile.
    """

    result: Dict[str, Any] = {}
    for output_name, field in (
        ("forward_speed_scale_mps", "forward"),
        ("lateral_speed_scale_mps", "lateral"),
        ("angular_speed_scale_radps", "angular"),
    ):
        real_data = np.abs(np.frombuffer(real[field], dtype=np.float32))
        sim_data = np.abs(np.frombuffer(sim[field], dtype=np.float32))
        field_result = {}
        for label, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            real_q = float(np.quantile(real_data, q))
            sim_q = float(np.quantile(sim_data, q))
            field_result[label] = {
                "real_abs_velocity": real_q,
                "sim_abs_normalized_action": sim_q,
                "real_speed_scale": real_q / sim_q if sim_q > 1e-12 else None,
            }
        result[output_name] = field_result
    result["recommended_quantile"] = "p95"
    return result


def write_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows = []
    datasets = {"real_pose_body": report["real"]["pose_derived_robot_frame"]}
    datasets["sim_combined_normalized"] = report["simulation"]["combined_normalized_action"]
    datasets["sim_combined_label_clock_equivalent"] = report["simulation"]["combined_label_clock_equivalent_metric_velocity"]
    for source, source_report in report["simulation"]["sources"].items():
        datasets[f"sim_{source}_normalized"] = source_report["normalized_action"]
        datasets[f"sim_{source}_effective"] = source_report["effective_metric_velocity"]
    for dataset, summary in datasets.items():
        for metric, stats in summary["metrics"].items():
            if stats is not None:
                rows.append({"dataset": dataset, "count": summary["count"], "metric": metric, **stats})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.real_dataset is not None:
        real_report, real_values = audit_real_dataset(args)
    else:
        real_report, real_values = audit_real(args)
    sim_report, sim_values = audit_sim(args)
    report = {
        "method": {
            "real": "consecutive pose delta / timestamp delta, rotated world->robot using starting yaw",
            "simulation": "first action per sample; clip [-1,1], multiply our Habitat task speed coefficient, integrate one 1/40 s env step, then divide by the dataset builder's synthetic 0.1 s label interval",
            "record_dt": args.record_dt,
            "integration_dt": args.integration_dt,
        },
        "real": real_report,
        "simulation": sim_report,
        "real_to_sim_normalization": normalization_scale_candidates(
            real_values, sim_values
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_csv, report)
    print(f"[done] JSON: {args.output_json}", flush=True)
    print(f"[done] CSV:  {args.output_csv}", flush=True)
    print(
        json.dumps(
            report["real_to_sim_normalization"], indent=2, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
