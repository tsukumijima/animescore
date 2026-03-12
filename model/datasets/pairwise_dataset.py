# datasets/pairwise_dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def _norm_relpath(p: str) -> str:
    p = str(p).strip().replace("\\", "/")
    # allow both "audio/0001.wav" and "0001.wav"
    if p.startswith("./"):
        p = p[2:]
    return p

class PairwiseMosDataset(Dataset):
    def __init__(self, csv_path, wav_root, target_sr=16000, max_sec=None):
        self.wav_root = wav_root
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_sec) if max_sec else None
        self.pairs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                a, b, choice = line.strip().split(",")
                choice = float(choice)
                y = 1.0 if choice == -1 else 0.0  # y=1 if a>b
                self.pairs.append((_norm_relpath(a), _norm_relpath(b), y))

    def _resolve(self, relpath: str) -> str:
        # Try direct join
        p = os.path.join(self.wav_root, relpath)
        if os.path.exists(p):
            return p
        # If relpath has "audio/" but wav_root already points to "audio", strip it
        # if relpath.startswith("audio/"):
        #     p2 = os.path.join(self.wav_root, relpath.replace("audio/", "", 1))
        #     if os.path.exists(p2):
        #         return p2
        return p  # fallback (will error and show path)

    def load_wav(self, relpath):
        path = self._resolve(relpath)
        wav, sr = torchaudio.load(path)
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
    wa, wb, y = zip(*batch)
    wa = pad_sequence(wa, batch_first=True)
    wb = pad_sequence(wb, batch_first=True)
    y = torch.stack(y)
    return wa, wb, y

def collate_pairwise(batch):
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