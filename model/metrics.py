"""
Utterance-level metrics and audio loading utility.
Shared by training and evaluation scripts.
"""

import numpy as np
import scipy.stats as st
import torch

from animescore._audio import load_audio


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2))


def lcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    r, _ = st.pearsonr(y_true_array, y_pred_array)
    if np.isnan(r) is True:
        return 0.0
    return float(r)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    r, _ = st.spearmanr(y_true_array, y_pred_array)
    if np.isnan(r) is True:
        return 0.0
    return float(r)


def ktau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    r, _ = st.kendalltau(y_true_array, y_pred_array)
    if np.isnan(r) is True:
        return 0.0
    return float(r)


def load_wav(path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    """Load and resample a waveform to the target sample rate."""

    return load_audio(path, target_sr).to(device)
