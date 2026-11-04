#!/usr/bin/env python3

"""Create reversed TrackVLA training episodes without changing the originals."""

import argparse
import copy
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MODES = ("AT", "DT", "STT")
ROBOT_MIN_START_DISTANCE = 0.9
ROBOT_MAX_START_DISTANCE = 1.5
ROBOT_MIN_NAVMESH_CLEARANCE = 0.1
ROBOT_ESCAPE_DISTANCE = 0.5
ROBOT_MIN_ESCAPE_DIRECTIONS = 2


def _reverse_episode(episode):
    old_main_start = list(episode["start_position"])
    main_goals = episode["goals"][0]["position"]
    main_route = [old_main_start] + [list(point) for point in main_goals]

    episode["start_position"] = main_route[-1]
    episode["goals"][0]["position"] = list(reversed(main_route[:-1]))

    info = episode["info"]
    info["ensure_robot_start_movable"] = True
    old_robot = info.get("robot_position")
    if old_robot is not None:
        # Preserve the direction of the original offset while constraining its
        # horizontal distance for a usable reversed tracking start.
        offset_x = old_robot[0] - old_main_start[0]
        offset_z = old_robot[2] - old_main_start[2]
        offset_distance = math.hypot(offset_x, offset_z)
        if offset_distance < 1e-8:
            offset_x = 1.0
            offset_z = 0.0
            offset_distance = 1.0
        desired_distance = min(
            ROBOT_MAX_START_DISTANCE,
            max(ROBOT_MIN_START_DISTANCE, offset_distance),
        )
        scale = desired_distance / offset_distance
        info["robot_position"] = [
            episode["start_position"][0] + offset_x * scale,
            episode["start_position"][1] + old_robot[1] - old_main_start[1],
            episode["start_position"][2] + offset_z * scale,
        ]

    waypoint_count = int(info["num_goals_for_other_human"])
    for human_index in range(1, int(info["human_num"]) + 1):
        start_key = f"human_{human_index}_start_position"
        if start_key not in info:
            continue

        waypoint_keys = [
            f"human_{human_index}_waypoint_{index}_position"
            for index in range(1, waypoint_count + 1)
        ]
        route = [list(info[start_key])] + [list(info[key]) for key in waypoint_keys]
        info[start_key] = route[-1]
        for key, point in zip(waypoint_keys, reversed(route[:-1])):
            info[key] = point


def _navmesh_path(scene_root, scene_id):
    scene_path = Path(scene_id)
    relative_path = scene_path.with_suffix("")
    return scene_root / relative_path.parent / f"{relative_path.name}.navmesh"


def _finite_point(point):
    values = [float(value) for value in point]
    return values if all(math.isfinite(value) for value in values) else None


def _rotation_toward(robot_position, target_position):
    dx = float(target_position[0]) - float(robot_position[0])
    dz = float(target_position[2]) - float(robot_position[2])
    return math.atan2(-dz, dx)


def _has_local_mobility(pathfinder, point):
    point_array = np.asarray(point, dtype=np.float32)
    if (
        pathfinder.distance_to_closest_obstacle(point_array, 2.0)
        < ROBOT_MIN_NAVMESH_CLEARANCE
    ):
        return False

    escape_directions = 0
    for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        target = point_array + np.array(
            [
                math.cos(angle) * ROBOT_ESCAPE_DISTANCE,
                0.0,
                math.sin(angle) * ROBOT_ESCAPE_DISTANCE,
            ],
            dtype=np.float32,
        )
        endpoint = np.asarray(
            pathfinder.try_step_no_sliding(point_array, target), dtype=np.float32
        )
        if np.linalg.norm((endpoint - target)[[0, 2]]) <= 0.05:
            escape_directions += 1
    return escape_directions >= ROBOT_MIN_ESCAPE_DIRECTIONS


def _snap_robot_start(pathfinder, candidate, main_start):
    snapped_main = _finite_point(pathfinder.snap_point(main_start))
    if snapped_main is None:
        raise RuntimeError(f"Main-human start is outside the NavMesh: {main_start}")

    island_index = pathfinder.get_island(snapped_main)
    snapped = _finite_point(pathfinder.snap_point(candidate, island_index))
    offset_x = candidate[0] - main_start[0]
    offset_z = candidate[2] - main_start[2]
    target_distance = math.hypot(offset_x, offset_z)
    angle = math.atan2(offset_z, offset_x)
    candidates = []

    if snapped is not None:
        distance = math.hypot(
            snapped[0] - snapped_main[0], snapped[2] - snapped_main[2]
        )
        if (
            ROBOT_MIN_START_DISTANCE <= distance <= ROBOT_MAX_START_DISTANCE
            and _has_local_mobility(pathfinder, snapped)
        ):
            score = abs(distance - target_distance) + 0.25 * math.hypot(
                snapped[0] - candidate[0], snapped[2] - candidate[2]
            )
            candidates.append((score, snapped))

    # Try the original direction first, then nearby directions at shorter radii.
    for radius_scale in (1.0, 0.9, 0.8, 0.7, 0.6):
        radius = max(ROBOT_MIN_START_DISTANCE, target_distance * radius_scale)
        for angle_offset in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05):
            probe_angle = angle + angle_offset
            probe = [
                main_start[0] + radius * math.cos(probe_angle),
                main_start[1],
                main_start[2] + radius * math.sin(probe_angle),
            ]
            point = _finite_point(pathfinder.snap_point(probe, island_index))
            if point is None:
                continue
            distance = math.hypot(
                point[0] - snapped_main[0], point[2] - snapped_main[2]
            )
            if (
                ROBOT_MIN_START_DISTANCE <= distance <= ROBOT_MAX_START_DISTANCE
                and _has_local_mobility(pathfinder, point)
            ):
                score = abs(distance - target_distance) + 0.25 * math.hypot(
                    point[0] - probe[0], point[2] - probe[2]
                )
                candidates.append((score, point))

    # Narrow corridors may not contain any of the deterministic probes.
    if not candidates:
        center = np.asarray(snapped_main, dtype=np.float32)
        for _ in range(128):
            point = _finite_point(
                pathfinder.get_random_navigable_point_near(
                    center, ROBOT_MAX_START_DISTANCE, 100, island_index
                )
            )
            if point is None:
                continue
            distance = math.hypot(
                point[0] - snapped_main[0], point[2] - snapped_main[2]
            )
            if (
                ROBOT_MIN_START_DISTANCE <= distance <= ROBOT_MAX_START_DISTANCE
                and _has_local_mobility(pathfinder, point)
            ):
                score = abs(distance - target_distance) + 0.25 * math.hypot(
                    point[0] - candidate[0], point[2] - candidate[2]
                )
                candidates.append((score, point))

    if not candidates:
        raise RuntimeError(
            f"Could not find a robot start near main-human start {main_start}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def _snap_cache_key(episode):
    return (
        episode["scene_id"],
        tuple(episode["start_position"]),
        tuple(episode["info"]["robot_position"]),
    )


def _snap_robot_starts(episodes, scene_root, snapped_positions):
    try:
        import habitat_sim
    except ImportError as error:
        raise RuntimeError(
            "habitat_sim is required to snap robot starts to the NavMesh"
        ) from error

    episodes_by_scene = defaultdict(list)
    for episode in episodes:
        episodes_by_scene[episode["scene_id"]].append(episode)

    pathfinder = habitat_sim.PathFinder()
    for scene_id, scene_episodes in episodes_by_scene.items():
        uncached = [
            episode
            for episode in scene_episodes
            if _snap_cache_key(episode) not in snapped_positions
        ]
        if uncached:
            navmesh_path = _navmesh_path(scene_root, scene_id)
            if not navmesh_path.is_file():
                raise FileNotFoundError(f"Missing NavMesh: {navmesh_path}")
            if not pathfinder.load_nav_mesh(str(navmesh_path)):
                raise RuntimeError(f"Could not load NavMesh: {navmesh_path}")
            pathfinder.seed(0)

            for episode in uncached:
                candidate = episode["info"]["robot_position"]
                snapped_list = _snap_robot_start(
                    pathfinder, candidate, episode["start_position"]
                )
                snapped_positions[_snap_cache_key(episode)] = snapped_list

        for episode in scene_episodes:
            episode["info"]["robot_position"] = snapped_positions[
                _snap_cache_key(episode)
            ]
            episode["info"]["robot_rotation"] = _rotation_toward(
                episode["info"]["robot_position"], episode["start_position"]
            )


def _write_json_gz(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, separators=(",", ":"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets/track"),
        help="Track dataset root (default: data/datasets/track)",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("data/scene_datasets"),
        help="Scene dataset root containing HM3D NavMeshes",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Dataset modes to generate",
    )
    parser.add_argument(
        "--output-suffix",
        default="_REVERSED",
        help="Suffix added to each output mode directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapped_positions = {}

    for mode in args.modes:
        input_path = args.data_root / mode / "train" / "train.json.gz"
        with gzip.open(input_path, "rt", encoding="utf-8") as input_file:
            source_dataset = json.load(input_file)

        reversed_dataset = copy.deepcopy(source_dataset)
        episodes = reversed_dataset["episodes"]
        for episode in episodes:
            _reverse_episode(episode)
        _snap_robot_starts(episodes, args.scene_root, snapped_positions)

        output_path = (
            args.data_root
            / f"{mode}{args.output_suffix}"
            / "train"
            / "train.json.gz"
        )
        _write_json_gz(output_path, reversed_dataset)
        print(f"Wrote {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
