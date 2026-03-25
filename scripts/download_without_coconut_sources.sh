#!/usr/bin/env bash
# Download only the source dataset files required for the without-coconut flow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_BIN="${HF_BIN:-hf}"
MAX_WORKERS="${MAX_WORKERS:-8}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
REAZONSPEECH_BASE_URL="${REAZONSPEECH_BASE_URL:-https://corpus.reazon-research.org/reazonspeech-v2}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "hf command was not found. Please install huggingface_hub and log in with \"hf auth login\"." >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT/downloads/anim400k/anim400k_audio_clips"
mkdir -p "$CACHE_ROOT/downloads/reazonspeech/data"
mkdir -p "$CACHE_ROOT/downloads/reazonspeech/tsv"

"$HF_BIN" download davidchan/anim400k \
  --repo-type dataset \
  --local-dir "$CACHE_ROOT/downloads/anim400k" \
  --max-workers "$MAX_WORKERS" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-00" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-01" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-02" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-03" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-04" \
  --include "anim400k_audio_clips/anim400k_audio_clips.tar.gz.part-05"

download_reazonspeech_file() {
  local relative_path="$1"
  local destination_path="$CACHE_ROOT/downloads/reazonspeech/$relative_path"
  local source_url="$REAZONSPEECH_BASE_URL/$relative_path"

  if [ -f "$destination_path" ]; then
    return 0
  fi

  mkdir -p "$(dirname "$destination_path")"
  curl \
    --fail \
    --location \
    --retry 3 \
    --continue-at - \
    --output "$destination_path" \
    "$source_url"
}

download_reazonspeech_file "data/000.tar"
download_reazonspeech_file "data/001.tar"
download_reazonspeech_file "data/002.tar"
download_reazonspeech_file "data/003.tar"
download_reazonspeech_file "tsv/small.tsv"
