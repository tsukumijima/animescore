# Data

```
data/
├── utterance_set/   # Per-utterance metadata (train / test splits)
├── pairs/           # A/B preference pair annotations
└── annotator/       # Annotator profiles and free-form descriptions
```

Audio files are **not** included. See [`../dataset/README.md`](../dataset/README.md) to download and preprocess the source corpora.

Filenames in all CSVs use anonymized shuffled IDs (e.g., `audio/0001.wav`) that are decoupled from the original corpus naming. The mapping to original corpus files is provided in the `original_file` column of the metadata CSVs.
