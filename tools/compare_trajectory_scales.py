#!/usr/bin/env python3
"""Compare real and simulation TrackVLA trajectory scales in metres.

Inputs can be TrackVLA roots, JSONL directories, individual JSONL files, or a
``dataset.json`` top-level array. A root containing ``dataset.json`` uses that
full manifest in preference to its smaller JSONL export. The JSON array reader
streams one object at a time, so large simulation manifests are not loaded
into RAM wholesale.

Every accepted trajectory is truncated to the first ``--waypoints`` positions.
This keeps the real and simulation horizon identical before scale statistics
are calculated.

Pass ``--per-waypoint-output-dir`` to also write one exact-waypoint report for
each position from 1 through ``--waypoints``. These reports do not resample
the trajectory: ``waypoint_03`` describes the third stored waypoint.

Example:
    ../envs/embodvla/bin/python tools/compare_trajectory_scales.py \
      --real data/dobot-trackvla-2 \
      --sim data/trackvla-data/at_train_60 \
      --waypoints 8 \
      --output outputs/real_vs_sim_trajectory_scale_8wp.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

QUANTILES = (0.05, 0.50, 0.95)
HORIZON_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SCALE_METRICS = (
    "endpoint_distance_m",
    "path_length_m",
    "max_forward_m",
    "max_abs_lateral_m",
    "final_abs_lateral_m",
)
WAYPOINT_METRICS = (
    "forward_m",
    "lateral_m",
    "abs_lateral_m",
    "distance_from_origin_m",
    "step_distance_m",
    "cumulative_path_length_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real and simulation TrackVLA trajectory scales in metres."
    )
    parser.add_argument(
        "--real",
        required=True,
        help="Real-data TrackVLA root, manifest directory, JSONL, or dataset.json.",
    )
    parser.add_argument(
        "--sim",
        action="append",
        required=True,
        help="Simulation root/directory/manifest. Repeat to combine sources.",
    )
    parser.add_argument(
        "--waypoints",
        type=int,
        default=8,
        help="Compare exactly the first N trajectory waypoints (default: 8).",
    )
    parser.add_argument(
        "--output",
        default="outputs/trajectory_scale_comparison.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional flat CSV report. Defaults to the JSON output with a .csv suffix.",
    )
    parser.add_argument(
        "--per-waypoint-output-dir",
        default=None,
        help=(
            "Write waypoint_01 ... waypoint_N JSON and CSV reports to this directory. "
            "Each report describes one exact stored waypoint."
        ),
    )
    return parser.parse_args()

def find_manifests(value: str) -> List[Path]:
    """Prefer a full dataset.json at a root, otherwise find JSONL manifests."""

    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix not in {".jsonl", ".json"}:
            raise ValueError(f"Expected a JSONL or JSON manifest, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    dataset_json = path / "dataset.json"
    if dataset_json.is_file():
        return [dataset_json]
    jsonl_root = path / "jsonl"
    search_root = jsonl_root if jsonl_root.is_dir() else path
    files = sorted(
        candidate
        for candidate in search_root.rglob("*.jsonl")
        if ".ipynb_checkpoints" not in candidate.parts
    )
    if not files:
        raise FileNotFoundError(f"No dataset.json or JSONL manifests found below {path}")
    return files


def iter_json_array_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream objects from a top-level JSON array without materialising it."""

    depth = 0
    in_string = False
    escaped = False
    object_bytes: bytearray | None = None
    saw_array = False

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            for byte in chunk:
                if depth > 0:
                    assert object_bytes is not None
                    object_bytes.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue
                if byte == ord('"') and depth > 0:
                    in_string = True
                elif byte == ord("[") and depth == 0:
                    saw_array = True
                elif byte == ord("{"):
                    if depth == 0:
                        object_bytes = bytearray((byte,))
                    depth += 1
                elif byte == ord("}") and depth > 0:
                    depth -= 1
                    if depth == 0:
                        assert object_bytes is not None
                        record = json.loads(object_bytes)
                        if not isinstance(record, dict):
                            raise ValueError(f"JSON array record in {path} is not an object")
                        yield record
                        object_bytes = None
    if not saw_array or depth != 0 or in_string or escaped:
        raise ValueError(f"Malformed top-level JSON array: {path}")


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"JSONL record in {path} is not an object")
                    yield record
        return
    if path.suffix == ".json":
        yield from iter_json_array_records(path)
        return
    raise ValueError(f"Unsupported manifest: {path}")


def load_trajectories(
    paths: Sequence[Path],
    waypoint_limit: int,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    trajectories: List[np.ndarray] = []
    counters = {
        "files": len(paths),
        "records": 0,
        "accepted": 0,
        "skipped_invalid": 0,
        "skipped_shorter_than_waypoint_limit": 0,
    }
    for path in paths:
        for record in iter_records(path):
            counters["records"] += 1
            try:
                trajectory = np.asarray(record["trajectory"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                counters["skipped_invalid"] += 1
                continue
            if (
                trajectory.ndim != 2
                or trajectory.shape[1] < 2
                or not np.isfinite(trajectory[:, :2]).all()
            ):
                counters["skipped_invalid"] += 1
                continue
            if len(trajectory) < waypoint_limit:
                counters["skipped_shorter_than_waypoint_limit"] += 1
                continue
            trajectories.append(trajectory[:waypoint_limit, :2])
            counters["accepted"] += 1
    if not trajectories:
        raise RuntimeError(f"No valid trajectories with {waypoint_limit} waypoints found")
    return trajectories, counters


def describe(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, QUANTILES[0])),
        "p50": float(np.quantile(values, QUANTILES[1])),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, QUANTILES[2])),
        "max": float(np.max(values)),
    }


def resample_with_origin(trajectory: np.ndarray) -> np.ndarray:
    points = np.vstack((np.zeros((1, 2), dtype=np.float64), trajectory))
    source = np.linspace(0.0, 1.0, num=len(points), dtype=np.float64)
    target = np.asarray(HORIZON_FRACTIONS, dtype=np.float64)
    return np.column_stack(
        (np.interp(target, source, points[:, 0]), np.interp(target, source, points[:, 1]))
    )


def summarise(trajectories: Sequence[np.ndarray], counters: Dict[str, int]) -> Dict[str, Any]:
    final_points = np.asarray([trajectory[-1] for trajectory in trajectories])
    endpoint_distance = np.linalg.norm(final_points, axis=1)
    path_lengths = np.asarray(
        [
            np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), trajectory)), axis=0), axis=1).sum()
            for trajectory in trajectories
        ]
    )
    max_forward = np.asarray([np.max(trajectory[:, 0]) for trajectory in trajectories])
    max_abs_lateral = np.asarray([np.max(np.abs(trajectory[:, 1])) for trajectory in trajectories])
    final_abs_lateral = np.abs(final_points[:, 1])
    normalised = np.asarray([resample_with_origin(trajectory) for trajectory in trajectories])

    metrics = {
        "endpoint_distance_m": describe(endpoint_distance),
        "path_length_m": describe(path_lengths),
        "max_forward_m": describe(max_forward),
        "max_abs_lateral_m": describe(max_abs_lateral),
        "final_forward_m": describe(final_points[:, 0]),
        "final_abs_lateral_m": describe(final_abs_lateral),
    }
    horizon_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for index, fraction in enumerate(HORIZON_FRACTIONS):
        point = normalised[:, index, :]
        horizon_metrics[f"{int(round(fraction * 100))}%"] = {
            "forward_m": describe(point[:, 0]),
            "abs_lateral_m": describe(np.abs(point[:, 1])),
            "distance_m": describe(np.linalg.norm(point, axis=1)),
        }
    return {"input": counters, "metrics": metrics, "normalised_horizon": horizon_metrics}



def summarise_waypoint(
    trajectories: Sequence[np.ndarray],
    waypoint_index: int,
    counters: Dict[str, int],
) -> Dict[str, Any]:
    """Describe one exact waypoint across all trajectories in metres."""

    if waypoint_index < 0:
        raise ValueError("waypoint_index must be non-negative")
    points = np.asarray([trajectory[waypoint_index] for trajectory in trajectories])
    previous_points = (
        np.zeros_like(points)
        if waypoint_index == 0
        else np.asarray([trajectory[waypoint_index - 1] for trajectory in trajectories])
    )
    cumulative_path_lengths = np.asarray(
        [
            np.linalg.norm(
                np.diff(
                    np.vstack((np.zeros((1, 2), dtype=np.float64), trajectory[: waypoint_index + 1])),
                    axis=0,
                ),
                axis=1,
            ).sum()
            for trajectory in trajectories
        ]
    )
    return {
        "input": counters,
        "metrics": {
            "forward_m": describe(points[:, 0]),
            "lateral_m": describe(points[:, 1]),
            "abs_lateral_m": describe(np.abs(points[:, 1])),
            "distance_from_origin_m": describe(np.linalg.norm(points, axis=1)),
            "step_distance_m": describe(np.linalg.norm(points - previous_points, axis=1)),
            "cumulative_path_length_m": describe(cumulative_path_lengths),
        },
    }

def ratio(simulation: float, real: float) -> float | None:
    return None if abs(real) < 1e-12 else float(simulation / real)


def scale_ratios(real: Dict[str, Any], simulation: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"global_metrics": {}, "normalised_horizon": {}}
    for metric in SCALE_METRICS:
        result["global_metrics"][metric] = {
            statistic: ratio(simulation["metrics"][metric][statistic], real["metrics"][metric][statistic])
            for statistic in ("p50", "p95")
        }
    for horizon, metrics in real["normalised_horizon"].items():
        result["normalised_horizon"][horizon] = {
            metric: {
                statistic: ratio(
                    simulation["normalised_horizon"][horizon][metric][statistic],
                    metrics[metric][statistic],
                )
                for statistic in ("p50", "p95")
            }
            for metric in ("forward_m", "abs_lateral_m", "distance_m")
        }
    return result



def waypoint_scale_ratios(real: Dict[str, Any], simulation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        metric: {
            statistic: ratio(simulation["metrics"][metric][statistic], real["metrics"][metric][statistic])
            for statistic in ("p50", "p95")
        }
        for metric in WAYPOINT_METRICS
    }

def write_csv(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group", "metric", "statistic", "value"))
        for group in ("real", "simulation"):
            summary = report[group]
            for metric, stats in summary["metrics"].items():
                for statistic, value in stats.items():
                    writer.writerow((group, metric, statistic, value))
            for horizon, metrics in summary["normalised_horizon"].items():
                for metric, stats in metrics.items():
                    for statistic, value in stats.items():
                        writer.writerow((group, f"horizon_{horizon}_{metric}", statistic, value))
        for metric, stats in report["simulation_to_real_ratio"]["global_metrics"].items():
            for statistic, value in stats.items():
                writer.writerow(("sim_to_real", metric, statistic, value))



def write_waypoint_csv(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group", "metric", "statistic", "value"))
        for group in ("real", "simulation"):
            for metric, stats in report[group]["metrics"].items():
                for statistic, value in stats.items():
                    writer.writerow((group, metric, statistic, value))
        for metric, stats in report["simulation_to_real_ratio"].items():
            for statistic, value in stats.items():
                writer.writerow(("sim_to_real", metric, statistic, value))

def print_summary(report: Dict[str, Any]) -> None:
    real = report["real"]
    simulation = report["simulation"]
    ratios = report["simulation_to_real_ratio"]["global_metrics"]
    print("\nTrajectory scale comparison (metres)")
    print(f"first {report['waypoints']} waypoints are used in both domains")
    print(f"real: {real['input']['accepted']} samples")
    print(f"sim:  {simulation['input']['accepted']} samples")
    print("metric                         real p50    sim p50    ratio    real p95    sim p95    ratio")
    for metric in SCALE_METRICS:
        real_stats = real["metrics"][metric]
        sim_stats = simulation["metrics"][metric]
        metric_ratio = ratios[metric]
        print(
            f"{metric:29s} {real_stats['p50']:9.3f} {sim_stats['p50']:10.3f} "
            f"{metric_ratio['p50']:7.3f} {real_stats['p95']:11.3f} {sim_stats['p95']:10.3f} "
            f"{metric_ratio['p95']:7.3f}"
        )


def write_per_waypoint_reports(
    output_dir: Path,
    *,
    waypoint_count: int,
    real_trajectories: Sequence[np.ndarray],
    real_counters: Dict[str, int],
    simulation_trajectories: Sequence[np.ndarray],
    simulation_counters: Dict[str, int],
    real_paths: Sequence[Path],
    simulation_paths: Sequence[Path],
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    for waypoint_index in range(waypoint_count):
        real = summarise_waypoint(real_trajectories, waypoint_index, real_counters)
        simulation = summarise_waypoint(simulation_trajectories, waypoint_index, simulation_counters)
        report = {
            "schema_version": 1,
            "waypoint_number": waypoint_index + 1,
            "waypoint_index_zero_based": waypoint_index,
            "real_inputs": [str(path) for path in real_paths],
            "simulation_inputs": [str(path) for path in simulation_paths],
            "real": real,
            "simulation": simulation,
            "simulation_to_real_ratio": waypoint_scale_ratios(real, simulation),
        }
        json_path = output_dir / f"waypoint_{waypoint_index + 1:02d}_scale_report.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_waypoint_csv(json_path.with_suffix(".csv"), report)
        output_paths.append(json_path)
    return output_paths

def main() -> None:
    args = parse_args()
    if args.waypoints <= 0:
        raise ValueError("--waypoints must be positive")
    real_paths = find_manifests(args.real)
    sim_paths = [path for value in args.sim for path in find_manifests(value)]
    real_trajectories, real_counters = load_trajectories(real_paths, args.waypoints)
    sim_trajectories, sim_counters = load_trajectories(sim_paths, args.waypoints)
    real = summarise(real_trajectories, real_counters)
    simulation = summarise(sim_trajectories, sim_counters)
    report = {
        "schema_version": 2,
        "waypoints": args.waypoints,
        "real_inputs": [str(path) for path in real_paths],
        "simulation_inputs": [str(path) for path in sim_paths],
        "real": real,
        "simulation": simulation,
        "simulation_to_real_ratio": scale_ratios(real, simulation),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = Path(args.csv_output).expanduser().resolve() if args.csv_output else output_path.with_suffix(".csv")
    write_csv(csv_path, report)
    print_summary(report)
    print(f"\nWrote JSON report: {output_path}")
    print(f"Wrote CSV report:  {csv_path}")
    if args.per_waypoint_output_dir:
        waypoint_output_dir = Path(args.per_waypoint_output_dir).expanduser().resolve()
        write_per_waypoint_reports(
            waypoint_output_dir,
            waypoint_count=args.waypoints,
            real_trajectories=real_trajectories,
            real_counters=real_counters,
            simulation_trajectories=sim_trajectories,
            simulation_counters=sim_counters,
            real_paths=real_paths,
            simulation_paths=sim_paths,
        )
        print(
            f"Wrote {args.waypoints} waypoint JSON/CSV report pairs under: "
            f"{waypoint_output_dir}"

        )

if __name__ == "__main__":
    main()
