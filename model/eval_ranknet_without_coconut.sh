#!/usr/bin/env bash
# Evaluate AnimeScore RankNet on the derived without-coconut split.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"

CKPT="${CKPT:-ckpt/animescore_without_coconut_hubert_best.pt}"
VAL_PAIR="${VAL_PAIR:-data/pairs/pair_eval_metadata_without_coconut.csv}"
UTT_CSV="${UTT_CSV:-data/utterance_set/pair_eval_utterance_metadata_without_coconut.csv}"
WAV_ROOT="${WAV_ROOT:-.}"

SSL_TYPE="${SSL_TYPE:-hubert}"
SSL_NAME="${SSL_NAME:-facebook/hubert-base-ls960}"

SAVE_PRED="${SAVE_PRED:-ckpt/predictions_without_coconut_${SSL_TYPE}.csv}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

"$UV_BIN" run python model/eval_ranknet_ckpt_v2.py \
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
