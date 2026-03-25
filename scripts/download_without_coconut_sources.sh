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

  mkdir -p "$(dirname "$destination_path")"
  curl \
    --fail \
    --location \
    --retry 3 \
    --retry-delay 5 \
    --max-time 7200 \
    --continue-at - \
    --output "$destination_path" \
    "$source_url"
}

export -f download_reazonspeech_file
export CACHE_ROOT
export REAZONSPEECH_BASE_URL

printf "%s\n" \
  "data/000.tar" \
  "data/001.tar" \
  "data/002.tar" \
  "data/003.tar" \
  "tsv/small.tsv" \
  | xargs -P "$MAX_WORKERS" -I {} bash -lc 'download_reazonspeech_file "$1"' _ "{}"
