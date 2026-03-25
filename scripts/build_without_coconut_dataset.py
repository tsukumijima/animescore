#!/usr/bin/env python3
"""
Build a canonical without-Coconut dataset from the published AnimeScore metadata.

This script resolves several inconsistencies in the public repository:

1. `shuffled_file` identifiers collide across the published train/test CSVs.
2. Pair CSVs include extra columns that the current training loader ignores.
3. `Coco-Nut` audio is unavailable in this environment.

The script creates a new without-Coconut dataset keyed by `original_file`,
assigns stable audio paths under `audio/`, filters out `Coco-Nut`, and writes
both official filtered splits and a merged re-split that can be used for
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


UTTERANCE_COLUMNS = [
    'shuffled_id',
    'shuffled_file',
    'original_file',
    'source',
    'ref_text',
    'cer',
    'duration_sec',
    'utmos',
]

PAIR_BASE_COLUMNS = [
    'file_a',
    'file_b',
    'choice',
    'source_a',
    'source_b',
    'original_file_a',
    'original_file_b',
]

PAIR_EXTRA_COLUMNS = [
    'speaker_cos',
    'text_cos',
]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo-root',
        default='.',
        help='Repository root that contains data/ and dataset/.',
    )
    parser.add_argument(
        '--cache-dir',
        default='.cache',
        help='Directory where summaries and temporary artifacts are written.',
    )
    parser.add_argument(
        '--eval-ratio',
        type=float,
        default=0.1,
        help='Evaluation ratio for the merged pair split.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used for the merged pair split.',
    )
    return parser.parse_args()


def normalize_source(source: str, original_file: str) -> str:
    """
    Normalize source labels that drifted in the public export.

    Args:
        source (str): Raw source label from a CSV row.
        original_file (str): Original corpus path associated with the row.

    Returns:
        str: Normalized source label.
    """

    source_value = source.strip()
    if source_value == 'animemos':
        if 'anim400k_audio_clips' in original_file:
            return 'anim400k'
    return source_value


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    Read a CSV file as dictionaries.

    Args:
        csv_path (Path): CSV file path.

    Returns:
        list[dict[str, str]]: Parsed rows.
    """

    with csv_path.open('r', encoding='utf-8', newline='') as file_pointer:
        return list(csv.DictReader(file_pointer))


def load_utterance_rows(repo_root: Path) -> dict[str, list[dict[str, str]]]:
    """
    Group utterance metadata rows by original file.

    Args:
        repo_root (Path): Repository root.

    Returns:
        dict[str, list[dict[str, str]]]: Rows grouped by `original_file`.
    """

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split_name in ('train', 'test'):
        csv_path = repo_root / 'data' / 'utterance_set' / f'{split_name}_metadata.csv'
        for row in read_csv_rows(csv_path):
            row['source'] = normalize_source(row['source'], row['original_file'])
            grouped_rows[row['original_file']].append(row)
    return grouped_rows


def load_without_coconut_pair_rows(repo_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Load pair rows after excluding `Coco-Nut`.

    Args:
        repo_root (Path): Repository root.

    Returns:
        tuple[list[dict[str, str]], list[dict[str, str]]]:
            Filtered official train rows and filtered official test rows.
    """

    filtered_rows: list[list[dict[str, str]]] = []
    for split_name in ('train', 'test'):
        csv_path = repo_root / 'data' / 'pairs' / f'pair_{split_name}_metadata.csv'
        current_rows: list[dict[str, str]] = []
        for row in read_csv_rows(csv_path):
            row['source_a'] = normalize_source(row['source_a'], row['original_file_a'])
            row['source_b'] = normalize_source(row['source_b'], row['original_file_b'])
            has_coco = row['source_a'] == 'coco_nut' or row['source_b'] == 'coco_nut'
            if has_coco is True:
                continue
            current_rows.append(row)
        filtered_rows.append(current_rows)
    return filtered_rows[0], filtered_rows[1]


def choose_canonical_utterance(rows: list[dict[str, str]]) -> dict[str, str]:
    """
    Choose a canonical utterance row for a duplicated original file.

    The public train/test exports often disagree only in punctuation. We keep a
    single canonical row per original file so that the restored audio has a
    stable destination.

    Args:
        rows (list[dict[str, str]]): Candidate rows for the same original file.

    Returns:
        dict[str, str]: Selected canonical row.
    """

    def row_priority(row: dict[str, str]) -> tuple[float, float, int, str]:
        cer_value = float(row['cer'])
        utmos_value = float(row['utmos'])
        text_length = len(row['ref_text'])
        return (cer_value, -utmos_value, -text_length, row['shuffled_file'])

    return sorted(rows, key=row_priority)[0]


def build_canonical_utterance_rows(
    grouped_utterances: dict[str, list[dict[str, str]]],
    pair_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    """
    Build canonical utterance rows referenced by the filtered pair annotations.

    Args:
        grouped_utterances (dict[str, list[dict[str, str]]]): Utterance rows grouped by original file.
        pair_rows (list[dict[str, str]]): Filtered pair rows.

    Returns:
        tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
            Canonical utterance rows, lookup by original file, and duplication stats.
    """

    required_originals: set[str] = set()
    for row in pair_rows:
        required_originals.add(row['original_file_a'])
        required_originals.add(row['original_file_b'])

    canonical_rows: list[dict[str, str]] = []
    canonical_lookup: dict[str, dict[str, str]] = {}
    duplication_counter = 0
    metadata_conflict_counter = 0

    for index, original_file in enumerate(sorted(required_originals), start=1):
        utterance_rows = grouped_utterances.get(original_file, [])
        if len(utterance_rows) == 0:
            raise FileNotFoundError(
                f'Missing utterance metadata for original file: {original_file}'
            )

        if len(utterance_rows) > 1:
            duplication_counter += 1
            signatures = {
                (
                    row['ref_text'],
                    row['cer'],
                    row['duration_sec'],
                    row['utmos'],
                )
                for row in utterance_rows
            }
            if len(signatures) > 1:
                metadata_conflict_counter += 1

        selected_row = choose_canonical_utterance(utterance_rows)
        canonical_row = {
            'shuffled_id': str(index),
            'shuffled_file': f'audio/{index:05d}.wav',
            'original_file': original_file,
            'source': selected_row['source'],
            'ref_text': selected_row['ref_text'],
            'cer': selected_row['cer'],
            'duration_sec': selected_row['duration_sec'],
            'utmos': selected_row['utmos'],
        }
        canonical_rows.append(canonical_row)
        canonical_lookup[original_file] = canonical_row

    return canonical_rows, canonical_lookup, {
        'duplicate_original_files': duplication_counter,
        'duplicates_with_metadata_conflict': metadata_conflict_counter,
    }


def remap_pair_rows(
    pair_rows: list[dict[str, str]],
    canonical_lookup: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """
    Replace public pair file identifiers with canonical without-Coconut audio paths.

    Args:
        pair_rows (list[dict[str, str]]): Filtered pair rows.
        canonical_lookup (dict[str, dict[str, str]]): Canonical utterance rows keyed by original file.

    Returns:
        list[dict[str, str]]: Remapped pair rows.
    """

    remapped_rows: list[dict[str, str]] = []
    for row in pair_rows:
        utterance_a = canonical_lookup[row['original_file_a']]
        utterance_b = canonical_lookup[row['original_file_b']]
        remapped_row = {
            'file_a': utterance_a['shuffled_file'],
            'file_b': utterance_b['shuffled_file'],
            'choice': row['choice'],
            'source_a': row['source_a'],
            'source_b': row['source_b'],
            'original_file_a': row['original_file_a'],
            'original_file_b': row['original_file_b'],
        }
        for extra_column in PAIR_EXTRA_COLUMNS:
            if extra_column in row:
                remapped_row[extra_column] = row[extra_column]
        remapped_rows.append(remapped_row)
    return remapped_rows


def write_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """
    Write dictionaries to a CSV file.

    Args:
        csv_path (Path): Output file path.
        fieldnames (list[str]): CSV header.
        rows (list[dict[str, str]]): Rows to write.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as file_pointer:
        writer = csv.DictWriter(file_pointer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_pairs_for_training(
    pair_rows: list[dict[str, str]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Split pair rows into train/eval while keeping label/source strata balanced.

    Args:
        pair_rows (list[dict[str, str]]): Pair rows to split.
        eval_ratio (float): Evaluation ratio in [0, 1).
        seed (int): Random seed.

    Returns:
        tuple[list[dict[str, str]], list[dict[str, str]]]:
            Train rows and evaluation rows.
    """

    random_generator = random.Random(seed)
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in pair_rows:
        stratum_key = '|'.join([
            row['source_a'],
            row['source_b'],
            row['choice'],
        ])
        strata[stratum_key].append(row)

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []

    for _, current_rows in sorted(strata.items()):
        shuffled_rows = list(current_rows)
        random_generator.shuffle(shuffled_rows)
        eval_count = int(round(len(shuffled_rows) * eval_ratio))
        if len(shuffled_rows) > 1 and eval_count == 0:
            eval_count = 1
        if eval_count >= len(shuffled_rows):
            eval_count = len(shuffled_rows) - 1
        eval_rows.extend(shuffled_rows[:eval_count])
        train_rows.extend(shuffled_rows[eval_count:])

    random_generator.shuffle(train_rows)
    random_generator.shuffle(eval_rows)
    return train_rows, eval_rows


def build_eval_utterance_rows(
    canonical_rows: list[dict[str, str]],
    eval_pair_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Select utterance rows referenced by the evaluation pairs.

    Args:
        canonical_rows (list[dict[str, str]]): Canonical utterance rows.
        eval_pair_rows (list[dict[str, str]]): Evaluation pair rows.

    Returns:
        list[dict[str, str]]: Evaluation utterance rows.
    """

    eval_audio_files: set[str] = set()
    for row in eval_pair_rows:
        eval_audio_files.add(row['file_a'])
        eval_audio_files.add(row['file_b'])

    return [
        row for row in canonical_rows
        if row['shuffled_file'] in eval_audio_files
    ]


def summarize_pairs(pair_rows: list[dict[str, str]]) -> dict[str, int]:
    """
    Summarize pair rows by ordered source tuple.

    Args:
        pair_rows (list[dict[str, str]]): Pair rows.

    Returns:
        dict[str, int]: Ordered source tuple counts.
    """

    counter: Counter[str] = Counter()
    for row in pair_rows:
        counter[f"{row['source_a']}->{row['source_b']}"] += 1
    return dict(sorted(counter.items()))


def main() -> None:
    """
    Build the without-Coconut dataset files.
    """

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    utterance_set_dir = repo_root / 'data' / 'utterance_set'
    pair_dir = repo_root / 'data' / 'pairs'

    official_train_pairs, official_test_pairs = load_without_coconut_pair_rows(repo_root)
    merged_pair_rows = official_train_pairs + official_test_pairs
    grouped_utterances = load_utterance_rows(repo_root)
    canonical_rows, canonical_lookup, duplication_stats = build_canonical_utterance_rows(
        grouped_utterances=grouped_utterances,
        pair_rows=merged_pair_rows,
    )

    remapped_official_train = remap_pair_rows(official_train_pairs, canonical_lookup)
    remapped_official_test = remap_pair_rows(official_test_pairs, canonical_lookup)
    remapped_all_pairs = remap_pair_rows(merged_pair_rows, canonical_lookup)

    merged_train_pairs, merged_eval_pairs = split_pairs_for_training(
        pair_rows=remapped_all_pairs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    eval_utterance_rows = build_eval_utterance_rows(canonical_rows, merged_eval_pairs)

    write_csv(
        utterance_set_dir / 'pair_pool_metadata_without_coconut.csv',
        UTTERANCE_COLUMNS,
        canonical_rows,
    )
    write_csv(
        utterance_set_dir / 'pair_eval_utterance_metadata_without_coconut.csv',
        UTTERANCE_COLUMNS,
        eval_utterance_rows,
    )
    write_csv(
        pair_dir / 'pair_train_metadata_without_coconut.csv',
        PAIR_BASE_COLUMNS + PAIR_EXTRA_COLUMNS,
        merged_train_pairs,
    )
    write_csv(
        pair_dir / 'pair_eval_metadata_without_coconut.csv',
        PAIR_BASE_COLUMNS + PAIR_EXTRA_COLUMNS,
        merged_eval_pairs,
    )
    summary = {
        'canonical_utterances': len(canonical_rows),
        'official_train_pairs': len(remapped_official_train),
        'official_test_pairs': len(remapped_official_test),
        'train_pairs': len(merged_train_pairs),
        'eval_pairs': len(merged_eval_pairs),
        'eval_utterances': len(eval_utterance_rows),
        'eval_ratio': args.eval_ratio,
        'seed': args.seed,
        'duplicate_original_files': duplication_stats['duplicate_original_files'],
        'duplicates_with_metadata_conflict': duplication_stats['duplicates_with_metadata_conflict'],
        'official_train_pair_sources': summarize_pairs(remapped_official_train),
        'official_test_pair_sources': summarize_pairs(remapped_official_test),
        'train_pair_sources': summarize_pairs(merged_train_pairs),
        'eval_pair_sources': summarize_pairs(merged_eval_pairs),
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cache_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(f'Wrote without-Coconut summary to: {summary_path}')
    print(f'Canonical utterances: {len(canonical_rows)}')
    print(f'Merged train pairs: {len(merged_train_pairs)}')
    print(f'Merged eval pairs: {len(merged_eval_pairs)}')
    print(f'Eval utterances: {len(eval_utterance_rows)}')
    print(
        'Duplicate original files resolved: '
        f"{duplication_stats['duplicate_original_files']}"
    )
    print(
        'Duplicate originals with metadata conflicts: '
        f"{duplication_stats['duplicates_with_metadata_conflict']}"
    )


if __name__ == '__main__':
    main()
