# AnimeScore (fork)

このフォークは、各種前提のずれや前処理不足を補ったうえで、**一般ユーザーが入手しやすいデータだけを使って AnimeScore を再学習・再利用しやすくすること**を目的とした派生実装です。

## Changes in this fork

- **`Coco-Nut` を使わない `without_coconut` 派生データセットを追加**
  - `Coco-Nut` は配布条件の都合で一般ユーザーが即時入手しづらいため、このフォークでは `Anim-400K` と `ReazonSpeech` に対応する公開済み 3,000 件サブセットのうち、`Coco-Nut` を除いた 1,968 発話で再構成した
  - 残る公開 pairwise アノテーションをできるだけ多く使うように派生 split を作成した
  - 利用できる規模は、発話数 1,968、pairwise アノテーション総数 6,813、学習用 pair 6,130、評価用 pair 683
- **公開 CSV と学習スクリプトの不整合を修正**
  - フォーク元のリポジトリでは、公開されている pair CSV の列構成と学習ローダーの前提が一致していなかった
  - このフォークでは、学習スクリプト側を現行の公開 CSV 形式に合わせて修正し、そのまま学習を回せるようにした
- **不足していた前処理スクリプトを追加し、再現用の手順を整理**
  - `scripts/download_without_coconut_sources.sh` を追加し、今回の再現フローで実際に使った `hf download` コマンドをそのまま再実行できるようにした
  - `scripts/build_without_coconut_dataset.py`、`scripts/prepare_without_coconut_audio.py`、`scripts/run_sidon_restore.py` を追加し、派生 split の構築、必要音声の抽出、Sidon による復元を段階的に実行できるようにした
  - 中間生成物とダウンロードキャッシュは `.cache/` 配下に集約し、最終的に本家が参照する `data/`、`dataset/`、`audio/` のみを公開レイアウトとして使うようにした
  - README も、上から順に実行すれば同じ流れを辿れるように整理した
- **学習済みモデルを他プロジェクトから使いやすいように Python パッケージ化**
  - `animescore` という Python パッケージを追加し、checkpoint を指定して推論できる API を用意した
  - 学習と推論の両方で使うモデル定義は `animescore` パッケージへ寄せ、学習スクリプト側は互換レイヤー越しに同じ実装を参照するようにした
  - 学習済み model は Hugging Face の `tsukumijima/animescore-without-coconut-hubert` へ公開しており、git から install したうえで checkpoint 未指定のまま Hugging Face 経由で利用できるようにしている
- **論文とは条件が異なるが、公開データだけでもそれなりの性能が出るところまで確認**
  - このフォークの学習条件は論文と同一ではないため、論文値との直接比較はできない
  - 現時点の最良 checkpoint は HuBERT ベースで、`without_coconut` の再分割 eval 683 pair に対して `Acc 0.904 / AUC 0.954`、official test 由来の filtered 1,228 pair に対して `Acc 0.845 / AUC 0.942` を確認している
  - したがって、少なくとも公開データだけでも実用上かなり強い pairwise scorer は作れていると考えられる

## Installation

学習や前処理まで含めて再現する場合は、リポジトリルートで次を実行します。

```bash
cd /path/to/animescore  # リポジトリルート
uv sync --python 3.11 --group train --no-managed-python
```

推論ライブラリとして他プロジェクトから使うだけであれば、git URL を直接指定して install できます。

```bash
uv pip install 'git+https://github.com/tsukumijima/animescore.git'
```

## Usage

デフォルトの Hugging Face モデルを自動取得して推論する最小例です。

```python
from animescore import AnimeScorePredictor

predictor = AnimeScorePredictor()

score = predictor.score_file('audio/00001.wav')
comparison = predictor.compare_files(
    'audio/00001.wav',
    'audio/00002.wav',
)
```

ローカル checkpoint を明示的に使いたい場合は、`checkpoint_path` を指定します。

```python
from animescore import AnimeScorePredictor

predictor = AnimeScorePredictor(
    checkpoint_path='ckpt/animescore_without_coconut_hubert_best.pt',
)

score = predictor.score_file('audio/00001.wav')
comparison = predictor.compare_files(
    'audio/00001.wav',
    'audio/00002.wav',
)
```

checkpoint を省略した場合は Hugging Face からの取得を優先します。  

## Documentation

- データ準備: [`dataset/README.md`](dataset/README.md)
- 学習と評価: [`model/README.md`](model/README.md)
- 推論 API: [`animescore/predictor.py`](animescore/predictor.py)

以下のドキュメントは、フォーク元の AnimeScore リポジトリのドキュメントを原則そのまま引き継いでいます。  
これらのドキュメントの内容がこのフォークにも通用するかは保証されません。

# AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style

> **Joonyong Park, Jerry Li**

AnimeScore addresses the lack of standardized metrics for evaluating anime-like speech quality. Because anime-likeness lacks a universal absolute scale, we adopt a **pairwise preference-based** approach: collect human A/B judgments, train a RankNet scorer on top of frozen SSL features, and anchor the resulting rankings to a MOS-like continuous scale.

---

## Repository Structure

```
animescore/
├── data/
│   ├── utterance_set/      # Per-utterance metadata (train / test splits)
│   ├── pairs/              # A/B preference pair annotations
│   └── annotator/          # Annotator profiles and free-form descriptions
├── model/                  # RankNet scorer (training + evaluation)
│   ├── src/                # Shared modules (SSL encoder, model, metrics)
│   └── datasets/           # Dataset loader
└── dataset/                # (Empty) Download & preprocessing instructions
```

Each subdirectory contains its own `README.md` with format details.

---

## Dataset Overview

### Speech Utterances

> [!NOTE]  
> The upstream README described this section as 3,000 utterances, but the metadata files currently published in this repository contain 2,999 rows in total: 2,499 in `train_metadata.csv` and 500 in `test_metadata.csv`.  
This fork updates the counts in the table below to match the published CSV files as they exist today.

The published metadata in this repository currently contains 2,999 utterances drawn from three publicly available Japanese speech corpora:

| Corpus | Train | Test | Total |
|--------|------:|-----:|------:|
| [Anim-400k](https://github.com/DavidMChan/Anim400K) | 1,064 | 250 | 1,314 |
| [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) | 828 | 120 | 948 |
| [Coco-Nut](https://arxiv.org/abs/2309.13509) | 607 | 130 | 737 |
| **Total** | **2,499** | **500** | **2,999** |

Utterances were filtered for quality and diversity:
- Duration: 2–10 s
- Character Error Rate (CER) < 0.30 (whisper-large-v3)
- UTMOS naturalness score > 3.0
- Speaker diversity via ECAPA-TDNN embedding clustering

Audio files are not included. See [`dataset/README.md`](dataset/README.md) for download and preprocessing instructions.

### A/B Preference Pairs

| Split | Pairs |
|-------|------:|
| Train | 12,500 |
| Test  |  2,500 |
| **Total** | **15,000** |

### Annotators

187 annotators recruited via Lancers (Japanese crowdsourcing platform).

| Category | Group | Count |
|----------|-------|------:|
| Age | 20s or younger | 8 |
| | 30s | 48 |
| | 40s | 80 |
| | 50s or older | 51 |
| Gender | Male | 142 |
| | Female | 45 |
| Anime Familiarity | Low | 9 |
| | Medium | 103 |
| | High | 75 |

---

## Model

### Architecture

```
Audio → Frozen SSL Encoder → BiLSTM → Mean Pooling → MLP → Score s(x) ∈ ℝ
```

Given a pair (a, b), the model predicts P(a ≻ b) = σ(s_a − s_b) and is trained with **RankNet** (pairwise logistic) loss. See [`model/README.md`](model/README.md) for training and evaluation instructions.

### Results

| Backbone | HuggingFace ID | Pairwise Acc. | AUC |
|----------|----------------|:-------------:|:---:|
| HuBERT | `facebook/hubert-base-ls960` | **82.4%** | **0.908** |
| WavLM | `microsoft/wavlm-base` | 81.1% | 0.894 |
| data2vec | `facebook/data2vec-audio-base-960h` | 77.1% | 0.858 |
| wav2vec 2.0 | `facebook/wav2vec2-base-960h` | 74.3% | 0.825 |

---

## License

Code: MIT License.

Audio data belongs to the respective original corpus licensors (Anim-400k, ReazonSpeech, Coco-Nut).
