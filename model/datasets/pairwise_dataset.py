"""Pairwise dataset loader for AnimeScore ranking experiments."""

import csv
from pathlib import Path

import soundfile
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def _norm_relpath(p: str) -> str:
    """Normalize a repository-relative audio path."""

    p = str(p).strip().replace("\\", "/")
    # allow both "audio/0001.wav" and "0001.wav"
    if p.startswith("./"):
        p = p[2:]
    return p


class PairwiseMosDataset(Dataset):
    """Load pairwise preference rows from the published CSV format."""

    def __init__(self, csv_path, wav_root, target_sr=16000, max_sec=None):
        self.wav_root = Path(wav_root)
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_sec) if max_sec else None
        self.pairs = []

        with open(csv_path, "r", encoding="utf-8", newline="") as file_pointer:
            reader = csv.DictReader(file_pointer)
            required_columns = {"file_a", "file_b", "choice"}
            if required_columns.issubset(set(reader.fieldnames or [])) is False:
                raise ValueError(
                    f"Pair CSV must include columns: {sorted(required_columns)}. "
                    f"Received: {reader.fieldnames}"
                )
            for row in reader:
                choice = float(row["choice"])
                # y=1 if a is preferred over b
                y = 1.0 if choice == 1.0 else 0.0
                self.pairs.append((
                    _norm_relpath(row["file_a"]),
                    _norm_relpath(row["file_b"]),
                    y,
                ))

    def _resolve(self, relpath: str) -> str:
        """Resolve a relative path against the configured audio root."""

        # Try direct join
        resolved_path = self.wav_root / relpath
        if resolved_path.exists() is True:
            return str(resolved_path)
        # fallback (will error and show path)
        return str(resolved_path)

    def load_wav(self, relpath):
        """Load and resample a waveform."""

        path = self._resolve(relpath)
        wav_array, sr = soundfile.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_array).transpose(0, 1)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.mean(0)  # [T]
        if self.max_len is not None and wav.numel() > self.max_len:
            wav = wav[: self.max_len]
        return wav

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        wav_a = self.load_wav(a)
        wav_b = self.load_wav(b)
        return wav_a, wav_b, torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)


def pairwise_collate(batch):
    """Pad pairwise waveforms to the batch maximum length."""

    wa, wb, y = zip(*batch)
    wa = pad_sequence(wa, batch_first=True)
    wb = pad_sequence(wb, batch_first=True)
    y = torch.stack(y)
    return wa, wb, y


def collate_pairwise(batch):
    """Collate a batch with explicit padding masks."""

    # pad to max length in batch
    wav_a, wav_b, y = zip(*batch)
    la = torch.tensor([t.numel() for t in wav_a], dtype=torch.long)
    lb = torch.tensor([t.numel() for t in wav_b], dtype=torch.long)
    maxa = int(la.max())
    maxb = int(lb.max())
    A = torch.stack([torch.nn.functional.pad(t, (0, maxa - t.numel())) for t in wav_a], dim=0)
    B = torch.stack([torch.nn.functional.pad(t, (0, maxb - t.numel())) for t in wav_b], dim=0)

    # attention masks (1 for real, 0 for pad)
    mask_a = torch.arange(maxa)[None, :] < la[:, None]
    mask_b = torch.arange(maxb)[None, :] < lb[:, None]
    y = torch.stack(y, dim=0)
    return A, mask_a, B, mask_b, y
