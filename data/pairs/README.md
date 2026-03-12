# A/B Preference Pairs

Pairwise preference judgments collected from 187 crowd annotators on Lancers.

## Files

### `pair_train_metadata.csv` — 12,500 training pairs

| Column | Description |
|--------|-------------|
| `file_a` | Shuffled audio path of utterance A |
| `file_b` | Shuffled audio path of utterance B |
| `choice` | Preference label: `1.0` = A preferred, `-1.0` = B preferred |
| `source_a` | Source corpus of A |
| `source_b` | Source corpus of B |
| `original_file_a` | Relative path to original corpus file for A |
| `original_file_b` | Relative path to original corpus file for B |

### `pair_test_metadata.csv` — 2,500 test pairs

Same columns as `pair_train_metadata.csv`, plus:

| Column | Description |
|--------|-------------|
| `speaker_cos` | ECAPA-TDNN speaker embedding cosine similarity between A and B |
| `text_cos` | Sentence embedding cosine similarity between A and B transcriptions |

## Construction

Pairs were sampled with the following priorities:

1. **Cross-corpus pairs** are weighted more heavily to minimize within-corpus acoustic homogeneity bias.
2. **Similarity-guided sampling**: pairs with similar transcriptions but different speakers are prioritized to reduce text and speaker confounds.
3. Each annotator judged approximately 80 pairs per session (5 sessions × 16 pairs).
4. The `choice` label reflects the **majority vote** across all annotators who saw the same pair.
