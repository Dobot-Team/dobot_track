#!/usr/bin/env python3
"""Aggregate one task/checkpoint using the evaluation controller's metric policy.

Usage: python tools/analyze_trackvla_bench.py --path <task_result_directory>
Only summary_split_*.json files directly inside that directory are considered.
Deduplicate by split_id, then weight SR/TR/CR by num_episodes. A shard's TR
is already the mean of its episode following rates, not a ratio of step sums.
For DT320 search, add --expected-splits 320 to select IDs 0..319.
The CLI writes following_info.json inside the result directory by default.
"""

import argparse
import json
import math
from numbers import Real
from pathlib import Path

METRICS = ('success_rate', 'following_rate', 'collision_rate')


def read_json_file(file_path):
    with Path(file_path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _validate_summary(data):
    if not isinstance(data, dict):
        raise ValueError('Expected a summary object')
    for key in ('split_id', 'split_num', 'num_episodes'):
        if type(data.get(key)) is not int:
            raise ValueError(f'{key} must be an integer')
    if data['split_num'] <= 0 or not 0 <= data['split_id'] < data['split_num']:
        raise ValueError('split_id must be in [0, split_num)')
    if data['num_episodes'] <= 0:
        raise ValueError('num_episodes must be positive')
    for key in METRICS:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{key} must be a number')
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'{key} must be finite and in [0, 1]')


def calculate_averages(folder_path, expected_splits=None):
    """Return controller-compatible metrics; never read episode/metadata JSONs.

    For retries or copied shards, the newest valid summary wins (mtime, with
    filename as a deterministic tie-breaker). Invalid or incomplete summaries
    are reported and cannot replace an older valid result. Point to one task,
    checkpoint and evaluation setting at a time, rather than a search root.
    """
    root = Path(folder_path)
    if not root.is_dir():
        raise ValueError(f'Not a result directory: {root}')
    rows, versions, duplicate_splits, skipped = {}, {}, set(), []
    split_counts = set()
    for path in sorted(root.glob('summary_split_*.json')):
        try:
            data = read_json_file(path)
            _validate_summary(data)
            version = (path.stat().st_mtime_ns, path.name)
        except (OSError, ValueError) as error:
            skipped.append({'file': path.name, 'reason': str(error)})
            continue
        split_id = data['split_id']
        split_counts.add(data['split_num'])
        if split_id in rows:
            duplicate_splits.add(split_id)
            if version <= versions[split_id]:
                continue
        rows[split_id] = data
        versions[split_id] = version
    if not rows:
        raise ValueError(f'No valid nonempty summary_split_*.json files found in {root}')
    if len(split_counts) != 1:
        raise ValueError('Mixed split_num values; analyze one task/evaluation run at a time')
    split_num = split_counts.pop()
    if expected_splits is None:
        expected_splits = split_num
    if type(expected_splits) is not int or not 0 < expected_splits <= split_num:
        raise ValueError('expected_splits must be an integer in [1, split_num]')
    rows = {i: row for i, row in rows.items() if i < expected_splits}
    if not rows:
        raise ValueError('No valid summaries in the requested split range')
    duplicate_splits.intersection_update(rows)
    count = sum(row['num_episodes'] for row in rows.values())
    result = {
        'completed_splits': len(rows),
        'split_num': split_num,
        'num_episodes': count,
        'expected_splits': expected_splits,
        'missing_splits': [i for i in range(expected_splits) if i not in rows],
    }
    for key in METRICS:
        result[key] = math.fsum(row[key] * row['num_episodes'] for row in rows.values()) / count
    result['duplicate_splits'] = sorted(duplicate_splits)
    result['skipped_summaries'] = skipped
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True, help='One task/checkpoint result directory')
    parser.add_argument('--expected-splits', type=int, help='Evaluate split IDs 0..N-1 (e.g. 320 for DT search)')
    parser.add_argument('--output', type=Path, help='Report path (default: <path>/following_info.json)')
    args = parser.parse_args()
    try:
        averages = calculate_averages(args.path, args.expected_splits)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or args.path / 'following_info.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(averages, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(output)
    print(text, end='')


if __name__ == '__main__':
    main()
