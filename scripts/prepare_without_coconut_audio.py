#!/usr/bin/env python3
"""
Extract the without-Coconut audio subset required by the published dataset layout.

This script performs two operations:

1. Stream-extract the required Anim-400K MP3 files from the multipart tarball.
2. Extract the required ReazonSpeech FLAC files from their shard tar archives and
   convert them into WAV files that match the repository layout.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
import tarfile
from pathlib import Path

import soundfile


class MultiPartReader(io.RawIOBase):
    """
    Read a multipart file sequence as one continuous binary stream.

    Args:
        part_paths (list[Path]): Ordered multipart file paths.
    """

    def __init__(self, part_paths: list[Path]) -> None:
        super().__init__()
        self._part_paths = part_paths
        self._part_index = 0
        self._current_handle = self._part_paths[0].open('rb')

    def readable(self) -> bool:
        """
        Report that the stream is readable.

        Returns:
            bool: Always `True`.
        """

        return True

    def readinto(self, buffer: bytearray) -> int:
        """
        Fill a writable buffer from the current multipart stream.

        Args:
            buffer (bytearray): Destination buffer.

        Returns:
            int: Number of bytes written. Zero signals EOF.
        """

        target_view = memoryview(buffer)
        total_bytes_read = 0

        while total_bytes_read < len(buffer):
            current_bytes = self._current_handle.readinto(target_view[total_bytes_read:])
            if current_bytes is None:
                current_bytes = 0
            if current_bytes > 0:
                total_bytes_read += current_bytes
                continue

            self._current_handle.close()
            self._part_index += 1
            if self._part_index >= len(self._part_paths):
                break
            self._current_handle = self._part_paths[self._part_index].open('rb')

        return total_bytes_read

    def close(self) -> None:
        """
        Close the active file handle.
        """

        if self._current_handle.closed is False:
            self._current_handle.close()
        super().close()


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
        help='Repository root that contains dataset/ and data/.',
    )
    parser.add_argument(
        '--utterance-csv',
        default='data/utterance_set/pair_pool_metadata_without_coconut.csv',
        help='Utterance metadata CSV created by build_without_coconut_dataset.py.',
    )
    parser.add_argument(
        '--cache-dir',
        default='.cache',
        help='Directory that stores raw downloads and manifests.',
    )
    return parser.parse_args()


def load_required_paths(
    utterance_csv: Path,
) -> tuple[set[str], dict[str, set[str]]]:
    """
    Load required Anim-400K members and ReazonSpeech WAV targets.

    Args:
        utterance_csv (Path): Derived utterance metadata CSV.

    Returns:
        tuple[set[str], dict[str, set[str]]]:
            Anim archive member names and Reazon WAV paths grouped by shard.
    """

    anim_members: set[str] = set()
    reazon_targets: dict[str, set[str]] = {}

    with utterance_csv.open('r', encoding='utf-8', newline='') as file_pointer:
        for row in csv.DictReader(file_pointer):
            original_file = row['original_file']
            if '/dataset/anim400k/' in original_file:
                member_name = original_file.split('/dataset/anim400k/', 1)[1]
                member_name = member_name.replace(
                    'anim400k_audio_clips/anim400k_audio_clips/',
                    'anim400k_audio_clips/',
                    1,
                )
                anim_members.add(member_name)
                continue
            if '/dataset/reazonspeech_wav_out/' in original_file:
                wav_relative_path = original_file.split('/dataset/reazonspeech_wav_out/', 1)[1]
                shard_name = wav_relative_path.split('/', 1)[0]
                if shard_name not in reazon_targets:
                    reazon_targets[shard_name] = set()
                reazon_targets[shard_name].add(wav_relative_path)
                continue
            raise ValueError(f'Unsupported original file: {original_file}')

    return anim_members, reazon_targets


def extract_anim400k_subset(
    repo_root: Path,
    cache_dir: Path,
    anim_members: set[str],
) -> int:
    """
    Stream-extract the required Anim-400K files from the multipart tarball.

    Args:
        repo_root (Path): Repository root.
        anim_members (set[str]): Required tar member names.

    Returns:
        int: Number of extracted files.
    """

    parts_dir = cache_dir / 'downloads' / 'anim400k' / 'anim400k_audio_clips'
    part_paths = sorted(parts_dir.glob('anim400k_audio_clips.tar.gz.part-*'))
    if len(part_paths) == 0:
        raise FileNotFoundError(f'Anim-400K multipart files were not found in: {parts_dir}')

    missing_members = set(anim_members)
    extracted_count = 0
    output_root = repo_root / 'dataset' / 'anim400k'

    output_root.mkdir(parents=True, exist_ok=True)

    reader = MultiPartReader(part_paths)
    gzip_reader = gzip.GzipFile(fileobj=reader, mode='rb')
    tar_reader = tarfile.open(fileobj=gzip_reader, mode='r|')

    for member in tar_reader:
        if member.isfile() is False:
            continue
        member_name = member.name.lstrip('./')
        if member_name not in missing_members:
            continue

        destination_path = output_root / member_name
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        extracted_stream = tar_reader.extractfile(member)
        if extracted_stream is None:
            raise FileNotFoundError(f'Failed to read archive member: {member_name}')
        with destination_path.open('wb') as output_file:
            shutil.copyfileobj(extracted_stream, output_file)

        missing_members.remove(member_name)
        extracted_count += 1
        if len(missing_members) == 0:
            break

    tar_reader.close()
    gzip_reader.close()
    reader.close()

    if len(missing_members) > 0:
        sample_members = sorted(missing_members)[:10]
        raise FileNotFoundError(
            'Some Anim-400K members were not found in the archive. '
            f'Missing count: {len(missing_members)}, sample: {sample_members}'
        )

    return extracted_count


def ensure_anim400k_compat_path(repo_root: Path) -> None:
    """
    Create a compatibility path expected by the published metadata.

    The downloaded tarball expands to:
        dataset/anim400k/anim400k_audio_clips/<prefix>/<uuid>.mp3

    The public AnimeScore metadata points to:
        dataset/anim400k/anim400k_audio_clips/anim400k_audio_clips/<prefix>/<uuid>.mp3

    We keep the extracted files in their native layout and add a symlink for the
    extra directory level so that both paths resolve correctly.

    Args:
        repo_root (Path): Repository root.
    """

    base_dir = repo_root / 'dataset' / 'anim400k' / 'anim400k_audio_clips'
    compat_dir = base_dir / 'anim400k_audio_clips'
    if compat_dir.exists() is True:
        return
    compat_dir.symlink_to(base_dir, target_is_directory=True)


def convert_flac_bytes_to_wav(flac_bytes: bytes, output_path: Path) -> None:
    """
    Convert FLAC bytes to a PCM WAV file.

    Args:
        flac_bytes (bytes): FLAC payload.
        output_path (Path): Destination WAV path.
    """

    audio_array, sample_rate = soundfile.read(io.BytesIO(flac_bytes), dtype='float32')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(output_path, audio_array, sample_rate, subtype='PCM_16')


def extract_reazonspeech_subset(
    repo_root: Path,
    cache_dir: Path,
    reazon_targets: dict[str, set[str]],
) -> int:
    """
    Extract and convert the required ReazonSpeech files.

    Args:
        repo_root (Path): Repository root.
        reazon_targets (dict[str, set[str]]): Required WAV paths grouped by shard.

    Returns:
        int: Number of converted WAV files.
    """

    reazon_archive_dir = cache_dir / 'downloads' / 'reazonspeech' / 'data'
    output_root = repo_root / 'dataset' / 'reazonspeech_wav_out'
    converted_count = 0

    for shard_name in sorted(reazon_targets):
        archive_path = reazon_archive_dir / f'{shard_name}.tar'
        if archive_path.exists() is False:
            raise FileNotFoundError(f'ReazonSpeech shard was not found: {archive_path}')

        with tarfile.open(archive_path, mode='r') as archive_handle:
            for wav_relative_path in sorted(reazon_targets[shard_name]):
                member_name = wav_relative_path.replace('.wav', '.flac')
                member = archive_handle.getmember(member_name)
                extracted_stream = archive_handle.extractfile(member)
                if extracted_stream is None:
                    raise FileNotFoundError(f'Failed to read archive member: {member_name}')
                flac_bytes = extracted_stream.read()
                output_path = output_root / wav_relative_path
                convert_flac_bytes_to_wav(flac_bytes, output_path)
                converted_count += 1

    return converted_count


def write_manifests(
    cache_dir: Path,
    anim_members: set[str],
    reazon_targets: dict[str, set[str]],
) -> None:
    """
    Write extraction manifests for reproducibility.

    Args:
        cache_dir (Path): Cache root.
        anim_members (set[str]): Required Anim-400K archive members.
        reazon_targets (dict[str, set[str]]): Required ReazonSpeech WAV paths.
    """

    manifests_dir = cache_dir / 'manifests'
    manifests_dir.mkdir(parents=True, exist_ok=True)

    anim_manifest = manifests_dir / 'anim400k_members.txt'
    anim_manifest.write_text(
        '\n'.join(sorted(anim_members)) + '\n',
        encoding='utf-8',
    )

    reazon_manifest = manifests_dir / 'reazonspeech_wav_paths.txt'
    all_reazon_paths: list[str] = []
    for current_paths in reazon_targets.values():
        all_reazon_paths.extend(sorted(current_paths))
    reazon_manifest.write_text(
        '\n'.join(all_reazon_paths) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    """
    Extract the without-Coconut audio subset.
    """

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    utterance_csv = (repo_root / args.utterance_csv).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()

    anim_members, reazon_targets = load_required_paths(utterance_csv)
    write_manifests(cache_dir, anim_members, reazon_targets)

    print(f'Preparing Anim-400K members: {len(anim_members)}')
    extracted_anim_count = extract_anim400k_subset(repo_root, cache_dir, anim_members)
    ensure_anim400k_compat_path(repo_root)
    print(f'Extracted Anim-400K files: {extracted_anim_count}')

    total_reazon_targets = sum(len(paths) for paths in reazon_targets.values())
    print(f'Preparing ReazonSpeech files: {total_reazon_targets}')
    converted_reazon_count = extract_reazonspeech_subset(
        repo_root,
        cache_dir,
        reazon_targets,
    )
    print(f'Converted ReazonSpeech files: {converted_reazon_count}')


if __name__ == '__main__':
    main()
