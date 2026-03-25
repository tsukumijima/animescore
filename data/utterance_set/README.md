# Utterance Set

> [!NOTE]  
> The original description referred to 3,000 utterances, but the metadata files currently published in this repository contain 2,999 rows in total: 2,499 in `train_metadata.csv` and 500 in `test_metadata.csv`.  
This README updates the counts below to match the published CSV files as they exist today.

Per-utterance metadata for the 2,999 utterances currently present in the published AnimeScore metadata files.

## Files

### `train_metadata.csv` — 2,499 training utterances

| Column | Description |
|--------|-------------|
| `shuffled_id` | Anonymized integer ID |
| `shuffled_file` | Shuffled audio path, e.g. `audio/0001.wav` |
| `original_file` | Relative path to the original corpus file under `dataset/` |
| `source` | Source corpus: `anim400k`, `reazonspeech`, `coco_nut` |
| `ref_text` | Reference Japanese transcription |
| `cer` | Character Error Rate vs. whisper-large-v3 ASR output |
| `duration_sec` | Audio duration in seconds |
| `utmos` | UTMOS naturalness score |

**Source distribution:**

| Source | Count |
|--------|------:|
| anim400k | 1,064 |
| reazonspeech | 828 |
| coco_nut | 607 |

### `test_metadata.csv` — 500 test utterances

Same schema as `train_metadata.csv`.

**Source distribution:**

| Source | Count |
|--------|------:|
| anim400k | 250 |
| reazonspeech | 120 |
| coco_nut | 130 |

## Notes

- `original_file` paths are relative to the repository root and assume the corpus audio has been downloaded and placed under `dataset/` as described in [`../../dataset/README.md`](../../dataset/README.md).
- `utmos` scores were computed using [UTMOS](https://github.com/tarepan/SpeechMOS) on the Sidon-restored audio.
- All utterances passed quality filters: duration 2–10 s, CER < 0.30, UTMOS > 3.0.
