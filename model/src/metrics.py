"""
Utterance-level metrics and audio loading utility.
Shared by training and evaluation scripts.
"""

import numpy as np
import torch
import scipy.stats as st


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2))


def lcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = st.pearsonr(y_true, y_pred)
    return float(r)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = st.spearmanr(y_true, y_pred)
    return float(r)


def ktau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = st.kendalltau(y_true, y_pred)
    return float(r)


def load_wav(path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).to(device)
