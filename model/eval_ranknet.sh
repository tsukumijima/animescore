#!/usr/bin/env bash
# Evaluate a saved AnimeScore RankNet checkpoint.
# Run from the repository root:
#   bash model/eval_ranknet.sh
set -euo pipefail

# ── Required: path to checkpoint ──────────────────────────────────────────────
CKPT="ckpt/animescore_hubert_best.pt"

# ── Data paths ─────────────────────────────────────────────────────────────────
VAL_PAIR="data/pairs/pair_test_metadata.csv"
UTT_CSV="data/utterance_set/test_metadata.csv"
WAV_ROOT="."

# ── SSL backbone (must match training) ────────────────────────────────────────
SSL_TYPE="hubert"                       # wav2vec2 | wavlm | hubert | data2vec | whisper-enc
SSL_NAME="facebook/hubert-base-ls960"

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_PRED="ckpt/predictions_${SSL_TYPE}.csv"
BATCH=8
NUM_WORKERS=2
DEVICE="cuda"
SEED=0

python model/eval_ranknet_ckpt_v2.py \
  --ckpt           "$CKPT" \
  --val_pair_csv   "$VAL_PAIR" \
  --wav_root       "$WAV_ROOT" \
  --ssl_type       "$SSL_TYPE" \
  --ssl_name       "$SSL_NAME" \
  --utt_csv        "$UTT_CSV" \
  --utt_wav_col    shuffled_file \
  --utt_score_col  utmos \
  --save_pred_csv  "$SAVE_PRED" \
  --batch_size     "$BATCH" \
  --num_workers    "$NUM_WORKERS" \
  --device         "$DEVICE" \
  --seed           "$SEED"
