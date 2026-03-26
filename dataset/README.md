# Dataset Preparation

Place downloaded and preprocessed audio files here following the structure
below, then run [`setup_dataset.sh`](setup_dataset.sh) to verify paths.

For the `without_coconut` flow supported by this fork, keep raw downloads under
`.cache/` and write the final extracted audio to the
same `dataset/` layout referenced by the metadata CSVs.

## Expected Structure

```
dataset/
├── anim400k/
│   └── anim400k_audio_clips/
│       └── anim400k_audio_clips/
│           ├── 00/{uuid}.mp3
│           ├── 01/{uuid}.mp3
│           └── ...             (subdirectory = first 2 chars of UUID)
├── reazonspeech_wav_out/
│   ├── 000/{name}.wav
│   ├── 001/{name}.wav
│   └── ...
└── coco_nut/
    └── wav/
        ├── 0/{name}.wav
        ├── 1/{name}.wav
        └── ...
```

The `original_file` columns in the metadata CSVs reference the **source** paths above.
After Sidon restoration (Step 2), the selected restored utterances are stored as `audio/{id}.wav`.

---

## Step 1 — Download

If you are reproducing the `without_coconut` flow in this fork, run:

```bash
cd /path/to/animescore  # repository root
uv sync --python 3.11 --group train
hf auth login

uv run python scripts/build_without_coconut_dataset.py \
  --repo-root . \
  --cache-dir .cache

bash scripts/download_without_coconut_sources.sh

uv run python scripts/prepare_without_coconut_audio.py \
  --repo-root . \
  --utterance-csv data/utterance_set/pair_pool_metadata_without_coconut.csv \
  --cache-dir .cache
```

This fork writes:

- raw downloads to `.cache/downloads/`
- final extracted `Anim-400K` audio to `dataset/anim400k/...`
- final extracted `ReazonSpeech` WAV files to `dataset/reazonspeech_wav_out/...`

### Anim-400k

Audio clips from Japanese anime (MP3, organized by UUID).

```bash
# Follow download instructions at https://github.com/DavidMChan/Anim400K
# Place audio under:
#   dataset/anim400k/anim400k_audio_clips/anim400k_audio_clips/{uuid[:2]}/{uuid}.mp3
```

### ReazonSpeech

Large-scale Japanese speech corpus.

```bash
# In this fork, raw downloads are cached under:
#   .cache/downloads/reazonspeech/
#
# Convert to WAV and organize as:
#   dataset/reazonspeech_wav_out/{subdir}/{name}.wav
```

See the [ReazonSpeech project page](https://research.reazon.jp/projects/ReazonSpeech/) for licensing.

### Coco-Nut

Japanese neutral speech corpus.

```bash
# Download from the release page linked in https://github.com/sarulab-speech/Coco-Nut
# Organize as:
#   dataset/coco_nut/wav/{subdir}/{name}.wav
```

---

## Step 2 — Speech Restoration with Sidon

All utterances were processed with **Sidon** (Nakata et al., ICASSP 2026), a neural speech restoration model that reduces background noise and recording artifacts.

> **Note:** Running Sidon over the full corpora takes a very long time. In this fork, run it only on the already selected utterances listed in `data/utterance_set/pair_pool_metadata_without_coconut.csv`.

For the `without_coconut` flow, restore the selected files with:

```bash
uv run python scripts/run_sidon_restore.py \
  --repo-root . \
  --utterance-csv data/utterance_set/pair_pool_metadata_without_coconut.csv \
  --device cuda \
  --batch-size 8 \
  --target-sample-rate 48000 \
  --skip-existing
```

If `--feature-extractor` and `--decoder` are omitted, the script downloads the
default TorchScript checkpoints from `sarulab-speech/sidon-v0.1`.

The restored files are written directly to `audio/{id}.wav`, matching the `shuffled_file` column.

---

## Step 3 — Quality Filtering (reference only)

The following filters were applied during dataset construction to arrive at the 3,000 utterances. **No action is needed here**; this section documents the criteria for reproducibility.

| Filter | Threshold | Tool |
|--------|-----------|------|
| Duration | 2–10 s | `ffprobe` / `torchaudio` |
| CER | < 0.30 | [whisper-large-v3](https://github.com/openai/whisper) |
| UTMOS | > 3.0 | [UTMOS](https://github.com/tarepan/SpeechMOS) |
| Text naturalness | ≤ 2 / 5 | Qwen3-30B-Instruct |

---

## Step 4 — Verify

Run the verification script from the repository root:

```bash
bash dataset/setup_dataset.sh --check
```

This checks that every `original_file` path listed in the metadata CSVs exists on disk.
