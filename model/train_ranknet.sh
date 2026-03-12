#!/usr/bin/env bash
# Train AnimeScore RankNet with a configurable SSL backbone.
# Run from the repository root:
#   bash model/train_ranknet.sh
set -euo pipefail

# ── Data paths (relative to repo root) ────────────────────────────────────────
TRAIN_PAIR="data/pairs/pair_train_metadata.csv"
VAL_PAIR="data/pairs/pair_test_metadata.csv"
UTT_CSV="data/utterance_set/test_metadata.csv"

# ── Audio root ─────────────────────────────────────────────────────────────────
# Point to the directory where restored audio files are stored.
# The pair CSVs reference paths like:
#   dataset/anim400k/anim400k_audio_clips/anim400k_audio_clips/87/876e...mp3
# so WAV_ROOT should be the repo root (or parent of 'dataset/').
WAV_ROOT="."

# ── SSL backbone ───────────────────────────────────────────────────────────────
SSL_TYPE="hubert"                       # wav2vec2 | wavlm | hubert | data2vec | whisper-enc
SSL_NAME="facebook/hubert-base-ls960"   # HuggingFace model ID

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS=20
BATCH=8
LR="3e-5"
NUM_WORKERS=2
DEVICE="cuda"
SEED=0

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_PATH="ckpt/animescore_${SSL_TYPE}.pt"
mkdir -p "$(dirname "$SAVE_PATH")"

python model/train_ranknet_v2.py \
  --train_pair_csv "$TRAIN_PAIR" \
  --val_pair_csv   "$VAL_PAIR" \
  --wav_root       "$WAV_ROOT" \
  --utt_csv        "$UTT_CSV" \
  --utt_wav_col    shuffled_file \
  --utt_score_col  utmos \
  --ssl_type       "$SSL_TYPE" \
  --ssl_name       "$SSL_NAME" \
  --freeze_ssl \
  --epochs         "$EPOCHS" \
  --batch_size     "$BATCH" \
  --lr             "$LR" \
  --num_workers    "$NUM_WORKERS" \
  --device         "$DEVICE" \
  --seed           "$SEED" \
  --save_path      "$SAVE_PATH"
