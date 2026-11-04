from astar_agent import evaluate_agent
import argparse
import copy
import habitat
from habitat.datasets import make_dataset
import evt_bench
import numpy as np
import random
import os
import json


def ensure_unique_episode_ids(dataset):
    """Prevent different trajectories from overwriting the same output file."""
    duplicate_counts = {}
    for episode in dataset.episodes:
        original_id = str(episode.episode_id)
        key = (episode.scene_id, original_id)
        duplicate_index = duplicate_counts.get(key, 0)
        duplicate_counts[key] = duplicate_index + 1
        if duplicate_index == 0:
            continue
        episode.info["source_episode_id"] = original_id
        episode.episode_id = f"{original_id}__dup{duplicate_index}"


def build_full_coverage_split(
    dataset, split_id, split_num, episodes_per_split, seed
):
    """Build deterministic chunks that include every episode exactly once."""
    episode_count = len(dataset.episodes)
    expected_splits = (
        episode_count + episodes_per_split - 1
    ) // episodes_per_split
    if split_num != expected_splits:
        raise ValueError(
            f"Full-coverage splitting needs {expected_splits} splits for "
            f"{episode_count} episodes at {episodes_per_split} per split, "
            f"got {split_num}"
        )

    random_indices = np.random.RandomState(seed).permutation(episode_count)
    indices_by_scene = {}
    for episode_index in random_indices:
        scene_id = dataset.episodes[int(episode_index)].scene_id
        indices_by_scene.setdefault(scene_id, []).append(int(episode_index))
    ordered_indices = [
        episode_index
        for scene_indices in indices_by_scene.values()
        for episode_index in scene_indices
    ]

    start = split_id * episodes_per_split
    stop = min(start + episodes_per_split, episode_count)
    dataset_split = copy.copy(dataset)
    dataset_split.episodes = [
        dataset.episodes[episode_index]
        for episode_index in ordered_indices[start:stop]
    ]
    return dataset_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["eval", "train"],
        required=True,
        help="run type",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-num",
        type=int,
        default=7,
        required=False,
    )

    parser.add_argument(
        "--episodes-per-split",
        type=int,
        default=None,
        help="Limit each split to this many episodes.",
    )

    parser.add_argument(
        "--reset-reserve-per-split",
        type=int,
        default=0,
        help="Extra episodes used when a target episode cannot be reset safely.",
    )

    parser.add_argument(
        "--full-coverage-splits",
        action="store_true",
        help="Cover every episode once and allow a shorter final split.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target-id",
        type=int,
        required=False,
        help="Semantic id of the specific person to follow (overrides name)",
    )

    parser.add_argument(
        "--target-name",
        type=str,
        required=False,
        help="Name of the person to follow, resolved via humanoid_infos.json",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    run_type: str,
    exp_config: str,
    split_id: int,
    split_num: int,
    episodes_per_split: int,
    reset_reserve_per_split: int,
    full_coverage_splits: bool,
    save_path: str,
    opts: None,
    target_id: int = None,
    target_name: str = None,
) -> None:
    config = habitat.get_config(exp_config, opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    ensure_unique_episode_ids(dataset)
    if not 0 <= split_id < split_num:
        raise ValueError(f"split_id must be in [0, {split_num}), got {split_id}")
    if episodes_per_split is not None and episodes_per_split <= 0:
        raise ValueError("episodes_per_split must be greater than zero")
    if reset_reserve_per_split < 0:
        raise ValueError("reset_reserve_per_split cannot be negative")

    if full_coverage_splits:
        if episodes_per_split is None:
            raise ValueError(
                "--full-coverage-splits requires --episodes-per-split"
            )
        if reset_reserve_per_split:
            raise ValueError(
                "Full-coverage splits cannot use reset reserve episodes"
            )
        dataset_split = build_full_coverage_split(
            dataset,
            split_id,
            split_num,
            episodes_per_split,
            config.habitat.simulator.seed,
        )
    else:
        dataset_splits = dataset.get_splits(
            split_num, episodes_per_split=episodes_per_split
        )
        if episodes_per_split is not None and reset_reserve_per_split:
            selected_objects = {
                id(episode)
                for current_split in dataset_splits
                for episode in current_split.episodes
            }
            remaining_episodes = [
                episode
                for episode in dataset.episodes
                if id(episode) not in selected_objects
            ]
            reserve_count = split_num * reset_reserve_per_split
            if reserve_count > len(remaining_episodes):
                raise ValueError(
                    f"Requested {reserve_count} reserve episodes, but only "
                    f"{len(remaining_episodes)} are available"
                )
            reserve_indices = np.random.choice(
                len(remaining_episodes), reserve_count, replace=False
            )
            for index, current_split in enumerate(dataset_splits):
                start = index * reset_reserve_per_split
                stop = start + reset_reserve_per_split
                current_split.episodes.extend(
                    remaining_episodes[item_index]
                    for item_index in reserve_indices[start:stop]
                )
        dataset_split = dataset_splits[split_id]

    target_episodes = len(dataset_split.episodes)
    print(
        f"Split {split_id}/{split_num}: collecting {target_episodes} episodes "
        f"from {len(dataset_split.episodes)} candidates"
    )

    # Resolve target human semantic id
    resolved_target_id = target_id
    if resolved_target_id is None and target_name:
        try:
            infos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "humanoid_infos.json"))
            with open(infos_path, "r") as f:
                infos = json.load(f)
            for entry in infos:
                if entry.get("name") == target_name:
                    resolved_target_id = int(entry.get("semantic_id"))
                    break
        except Exception:
            resolved_target_id = None

    if run_type == "eval":
        evaluate_agent(
            config,
            dataset_split,
            save_path,
            resolved_target_id,
            max_episodes=target_episodes,
        )
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
