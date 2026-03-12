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

3,000 utterances drawn from three publicly available Japanese speech corpora:

| Corpus | Train | Test | Total |
|--------|------:|-----:|------:|
| [Anim-400k](https://github.com/DavidMChan/Anim400K) | 1,065 | 250 | 1,315 |
| [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) | 828 | 120 | 948 |
| [Coco-Nut](https://arxiv.org/abs/2309.13509) | 607 | 130 | 737 |
| **Total** | **2,500** | **500** | **3,000** |

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
