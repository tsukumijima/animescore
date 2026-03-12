# Dataset Preparation

This directory is intentionally empty. Place downloaded and preprocessed audio files here following the structure below, then run [`setup_dataset.sh`](setup_dataset.sh) to verify paths.

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
After Sidon restoration (Step 2), the 3,000 restored utterances are stored as `audio/{id}.wav`.

---

## Step 1 — Download

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
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="reazon-research/reazonspeech",
    repo_type="dataset",
    local_dir="dataset/reazonspeech_raw",
)
EOF
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

> **Note:** Running Sidon over the full corpora takes a very long time. Instead, run it only on the **3,000 utterances already selected** in the metadata. Use the helper script to extract the required input paths:
>
> ```bash
> bash dataset/setup_dataset.sh --list-inputs > sidon_input_list.txt
> ```

Then restore each file:

```bash
# Install Sidon
git clone <sidon-repo-url>
cd sidon && pip install -r requirements.txt && cd ..

# Restore the 3,000 selected files (output to audio/)
mkdir -p audio
while IFS= read -r src; do
    id=$(basename "$src" | sed 's/\.[^.]*$//')
    python sidon/sidon_gpu.py \
      --input  "$src" \
      --output "audio/${id}_restored.wav" \
      --device cuda
done < sidon_input_list.txt
```

Rename or remap the restored files to match the `shuffled_file` column (`audio/{id}.wav`) using the `original_file` → `shuffled_file` mapping in `train_metadata.csv` / `test_metadata.csv`.

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
