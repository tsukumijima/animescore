#!/usr/bin/env bash
# Download only the source dataset files required for the without-coconut flow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_BIN="${HF_BIN:-hf}"
MAX_WORKERS="${MAX_WORKERS:-8}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "hf command was not found. Please install huggingface_hub and log in with 'hf auth login'." >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT/downloads/anim400k/anim400k_audio_clips"
mkdir -p "$CACHE_ROOT/downloads/reazonspeech"

"$HF_BIN" download davidchan/anim400k \
  --repo-type dataset \
  --local-dir "$CACHE_ROOT/downloads/anim400k" \
  --max-workers "$MAX_WORKERS" \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-00' \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-01' \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-02' \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-03' \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-04' \
  --include 'anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-05'

"$HF_BIN" download reazon-research/reazonspeech \
  --repo-type dataset \
  --local-dir "$CACHE_ROOT/downloads/reazonspeech" \
  --max-workers "$MAX_WORKERS" \
  --include 'data/000.tar' \
  --include 'data/001.tar' \
  --include 'data/002.tar' \
  --include 'data/003.tar' \
  --include 'tsv/small.tsv'
