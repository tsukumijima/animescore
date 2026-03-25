"""Audio loading helpers for AnimeScore inference."""

from __future__ import annotations

from pathlib import Path

import soundfile
import torch
import torch.nn.functional as functional
import torchaudio


def load_audio(audio_path: str | Path, target_sample_rate: int) -> torch.Tensor:
    """
    Load an audio file, convert it to mono, and resample it for the model.

    Args:
        audio_path (str | Path): Path to the input audio file.
        target_sample_rate (int): Target sample rate expected by the SSL encoder.

    Returns:
        torch.Tensor: One-dimensional waveform tensor on CPU in float32 format.
    """

    waveform_array, sample_rate = soundfile.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    waveform_tensor = torch.from_numpy(waveform_array).transpose(0, 1)

    if waveform_tensor.size(0) > 1:
        waveform_tensor = waveform_tensor.mean(dim=0, keepdim=True)

    if int(sample_rate) != target_sample_rate:
        waveform_tensor = torchaudio.functional.resample(
            waveform_tensor,
            int(sample_rate),
            target_sample_rate,
        )

    return waveform_tensor.squeeze(0)


def build_padded_batch(waveforms: list[torch.Tensor]) -> torch.Tensor:
    """
    Pad a waveform list to the maximum length and stack it into a batch tensor.

    Args:
        waveforms (list[torch.Tensor]): List of one-dimensional waveform tensors.

    Returns:
        torch.Tensor: Batch tensor of shape `[batch_size, max_num_samples]`.
    """

    if len(waveforms) == 0:
        raise ValueError(
            "build_padded_batch: 'waveforms' must be a non-empty list of 1D torch.Tensors.",
        )

    max_num_samples = max(waveform.numel() for waveform in waveforms)
    return torch.stack(
        [
            functional.pad(waveform, (0, max_num_samples - waveform.numel()))
            for waveform in waveforms
        ],
        dim=0,
    )
