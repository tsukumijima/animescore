#!/usr/bin/env bash
# Train AnimeScore RankNet on the derived without-coconut split.
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"

TRAIN_PAIR="${TRAIN_PAIR:-data/pairs/pair_train_metadata_without_coconut.csv}"
VAL_PAIR="${VAL_PAIR:-data/pairs/pair_eval_metadata_without_coconut.csv}"
UTT_CSV="${UTT_CSV:-data/utterance_set/pair_eval_utterance_metadata_without_coconut.csv}"
WAV_ROOT="${WAV_ROOT:-.}"

SSL_TYPE="${SSL_TYPE:-hubert}"
SSL_NAME="${SSL_NAME:-facebook/hubert-base-ls960}"

EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

SAVE_PATH="${SAVE_PATH:-ckpt/animescore_without_coconut_${SSL_TYPE}.pt}"
mkdir -p "$(dirname "$SAVE_PATH")"

"$UV_BIN" run python model/train_ranknet_v2.py \
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
