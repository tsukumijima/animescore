# Model

RankNet-based anime-likeness scorer using frozen SSL encoders.

## Architecture

```
Audio → Frozen SSL Encoder → BiLSTM → Mean Pooling → MLP → Score s(x) ∈ ℝ
```

Pairwise training loss (RankNet):
```
L = mean[ softplus( -y · (s_a - s_b) ) ]    y ∈ {-1, +1}
```

## Supported SSL Backbones

| `--ssl_type` | Default `--ssl_name` | Pairwise Acc. | AUC |
|---|---|:---:|:---:|
| `hubert` | `facebook/hubert-base-ls960` | **82.4%** | **0.908** |
| `wavlm` | `microsoft/wavlm-base` | 81.1% | 0.894 |
| `data2vec` | `facebook/data2vec-audio-base-960h` | 77.1% | 0.858 |
| `wav2vec2` | `facebook/wav2vec2-base-960h` | 74.3% | 0.825 |
| `whisper-enc` | `openai/whisper-small` | — | — |

## Directory Layout

```
model/
├── train_ranknet_v2.py       # Training script
├── eval_ranknet_ckpt_v2.py   # Evaluation script
├── train_ranknet.sh          # Training shell script (run from repo root)
├── eval_ranknet.sh           # Evaluation shell script (run from repo root)
├── requirements.txt
├── src/
│   ├── ssl_encoder.py        # SSL wrappers + freeze helpers
│   ├── ranknet_model.py      # RankNetMos
│   └── metrics.py            # mse / lcc / srcc / ktau / load_wav
└── datasets/
    └── pairwise_dataset.py   # PairwiseMosDataset + collate fn
```

## Installation

```bash
pip install -r model/requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0. CUDA recommended.

---

## Training

### Quick Start

Edit `WAV_ROOT` in `model/train_ranknet.sh`, then run from the **repository root**:

```bash
bash model/train_ranknet.sh
```

### Manual

```bash
python model/train_ranknet_v2.py \
  --train_pair_csv data/pairs/pair_train_metadata.csv \
  --val_pair_csv   data/pairs/pair_test_metadata.csv \
  --wav_root       . \
  --utt_csv        data/utterance_set/test_metadata.csv \
  --utt_wav_col    shuffled_file \
  --utt_score_col  utmos \
  --ssl_type       hubert \
  --ssl_name       facebook/hubert-base-ls960 \
  --freeze_ssl \
  --epochs         20 \
  --batch_size     8 \
  --save_path      ckpt/animescore_hubert.pt
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_pair_csv` | required | Training pairs CSV |
| `--val_pair_csv` | required | Validation/test pairs CSV |
| `--wav_root` | required | Root for resolving `original_file` paths in the pair CSVs |
| `--utt_csv` | — | CSV with per-utterance labels for utterance-level metrics |
| `--utt_wav_col` | `audio` | Column name for audio paths in `--utt_csv` |
| `--utt_score_col` | `score` | Column name for target scores in `--utt_csv` |
| `--ssl_type` | `wav2vec2` | SSL backbone type |
| `--ssl_name` | — | HuggingFace model ID (default per backbone if omitted) |
| `--freeze_ssl` | flag | Freeze SSL encoder (recommended) |
| `--unfreeze_top_n` | `0` | Unfreeze top-N encoder layers instead |
| `--save_path` | `mos_ranknet.pt` | Final checkpoint path |

Saves two checkpoints: `{save_path}` (final) and `{save_path}_best.pt` (best val loss).

---

## Evaluation

### Quick Start

Edit `CKPT` in `model/eval_ranknet.sh`, then run from the **repository root**:

```bash
bash model/eval_ranknet.sh
```

### Manual

```bash
python model/eval_ranknet_ckpt_v2.py \
  --ckpt           ckpt/animescore_hubert_best.pt \
  --val_pair_csv   data/pairs/pair_test_metadata.csv \
  --wav_root       . \
  --ssl_type       hubert \
  --utt_csv        data/utterance_set/test_metadata.csv \
  --utt_wav_col    shuffled_file \
  --utt_score_col  utmos \
  --save_pred_csv  ckpt/predictions_hubert.csv
```

### Output Metrics

**Primary (pairwise):**

| Metric | Description |
|--------|-------------|
| `pair_nll` | RankNet NLL (lower is better) |
| `pair_acc` | Pairwise accuracy |
| `pair_auc` | ROC-AUC using score margin s_a − s_b |

**Auxiliary (utterance-level, requires `--utt_csv`):**

| Metric | Description |
|--------|-------------|
| `MSE` | Mean squared error vs. target scores |
| `LCC` | Pearson linear correlation |
| `SRCC` | Spearman rank correlation |
| `KTAU` | Kendall's tau |

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | required | Path to saved `.pt` checkpoint |
| `--val_pair_csv` | required | Test pairs CSV |
| `--wav_root` | required | Audio root directory |
| `--ssl_type` | `wav2vec2` | Must match training |
| `--ssl_name` | — | Must match training |
| `--save_pred_csv` | — | Save per-utterance predictions to CSV |

---

## Audio File Resolution

`PairwiseMosDataset` resolves audio paths as `{wav_root}/{file_a}` (or `file_b`). The pair CSVs store `original_file` paths like:

```
./dataset/anim400k/anim400k_audio_clips/anim400k_audio_clips/87/876e...mp3
```

Set `--wav_root .` to use these as-is from the repository root, assuming `dataset/` has been populated per [`../dataset/README.md`](../dataset/README.md).
