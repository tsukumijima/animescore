#!/usr/bin/env bash
# setup_dataset.sh — utilities for dataset verification and Sidon input preparation.
#
# Usage (from repo root):
#   bash dataset/setup_dataset.sh                # check paths, print summary
#   bash dataset/setup_dataset.sh --check        # exit with error if any file is missing
#   bash dataset/setup_dataset.sh --list-inputs  # print unique original_file paths (for Sidon)
set -euo pipefail

MODE="${1:-}"

# --list-inputs: print unique source paths from train/test metadata (for Sidon input)
if [[ "$MODE" == "--list-inputs" ]]; then
    TRAIN_META="data/utterance_set/train_metadata.csv"
    TEST_META="data/utterance_set/test_metadata.csv"
    for csv in "$TRAIN_META" "$TEST_META"; do
        idx=$(head -1 "$csv" | tr ',' '\n' | grep -n "^original_file$" | cut -d: -f1)
        tail -n +2 "$csv" | cut -d, -f"$idx" | tr -d '"' | sed 's|^\./||'
    done | sort -u
    exit 0
fi

STRICT=false
[[ "$MODE" == "--check" ]] && STRICT=true

TRAIN_META="data/utterance_set/train_metadata.csv"
TEST_META="data/utterance_set/test_metadata.csv"
PAIR_TRAIN="data/pairs/pair_train_metadata.csv"
PAIR_TEST="data/pairs/pair_test_metadata.csv"

missing=0
total=0

check_col () {
    local csv="$1"
    local col="$2"
    local header
    header=$(head -1 "$csv")
    local idx
    idx=$(echo "$header" | tr ',' '\n' | grep -n "^${col}$" | cut -d: -f1)
    if [[ -z "$idx" ]]; then
        echo "[WARN] Column '${col}' not found in ${csv}" >&2
        return
    fi
    while IFS=, read -r -a fields; do
        local path="${fields[$((idx-1))]}"
        path="${path//\"/}"        # strip quotes
        path="${path#./}"          # strip leading ./
        [[ -z "$path" || "$path" == "$col" ]] && continue
        total=$((total + 1))
        if [[ ! -f "$path" ]]; then
            echo "MISSING: $path"
            missing=$((missing + 1))
        fi
    done < <(tail -n +2 "$csv")
}

echo "Checking utterance metadata..."
check_col "$TRAIN_META" "original_file"
check_col "$TEST_META"  "original_file"

echo "Checking pair metadata..."
check_col "$PAIR_TRAIN" "original_file_a"
check_col "$PAIR_TRAIN" "original_file_b"
check_col "$PAIR_TEST"  "original_file_a"
check_col "$PAIR_TEST"  "original_file_b"

echo ""
echo "Results: ${missing} missing / ${total} total files checked"

if [[ "$missing" -gt 0 ]]; then
    echo ""
    echo "Some files are missing. Follow dataset/README.md to download and restore the corpora."
    $STRICT && exit 1
else
    echo "All files present."
fi
