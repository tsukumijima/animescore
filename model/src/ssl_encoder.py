"""
SSL encoder wrappers shared by training and evaluation scripts.

Supported backbones:
  wav2vec2  : facebook/wav2vec2-base-960h
  hubert    : facebook/hubert-base-ls960
  wavlm     : microsoft/wavlm-base (or -large)
  data2vec  : facebook/data2vec-audio-base-960h
  whisper-enc: openai/whisper-small (encoder only)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoFeatureExtractor,
    WhisperModel,
)


@dataclass
class SSLSpec:
    ssl_type: str
    name_or_path: str
    target_sr: int
    feat_dim: int


class HFSSLWrapper(nn.Module):
    """Generic wrapper for Wav2Vec2 / HuBERT / WavLM / data2vec style models."""

    def __init__(self, name_or_path: str, output_hidden_states: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(name_or_path, output_hidden_states=output_hidden_states)
        self.model = AutoModel.from_pretrained(name_or_path, config=self.config)
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name_or_path)
        self.target_sr = getattr(self.feature_extractor, "sampling_rate", 16000)
        self.feat_dim = int(self.config.hidden_size)

    def forward(self, wav_16k: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out = self.model(input_values=wav_16k, attention_mask=attention_mask)
        return {
            "last_hidden_state": out.last_hidden_state,
            "hidden_states": getattr(out, "hidden_states", None),
        }

    @property
    def encoder_layers(self) -> List[nn.Module]:
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            return list(self.model.encoder.layers)
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return list(self.model.encoder.layer)
        return []


class WhisperEncWrapper(nn.Module):
    """Whisper encoder-only wrapper (log-mel → encoder hidden states)."""

    def __init__(self, name_or_path: str = "openai/whisper-small"):
        super().__init__()
        self.fe = AutoFeatureExtractor.from_pretrained(name_or_path)
        self.model = WhisperModel.from_pretrained(name_or_path)
        self.target_sr = getattr(self.fe, "sampling_rate", 16000)
        self.feat_dim = int(self.model.config.d_model)

    def forward(self, wav_16k: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        wav = wav_16k.detach().cpu().numpy()
        feats = self.fe(wav, sampling_rate=self.target_sr, return_tensors="pt")
        input_features = feats["input_features"].to(wav_16k.device)
        enc = self.model.encoder(input_features=input_features)
        return {"last_hidden_state": enc.last_hidden_state, "hidden_states": None}

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return list(self.model.encoder.layers)


DEFAULT_SSL_NAMES = {
    "wav2vec2":   "facebook/wav2vec2-base-960h",
    "hubert":     "facebook/hubert-base-ls960",
    "wavlm":      "microsoft/wavlm-base",
    "data2vec":   "facebook/data2vec-audio-base-960h",
    "whisper-enc": "openai/whisper-small",
}


def build_ssl(spec: SSLSpec) -> nn.Module:
    t = spec.ssl_type.lower()
    if t in ("wav2vec2", "hubert", "wavlm", "data2vec"):
        return HFSSLWrapper(spec.name_or_path, output_hidden_states=True)
    if t in ("whisper-enc", "whisper"):
        return WhisperEncWrapper(spec.name_or_path)
    raise ValueError(f"Unknown ssl_type={spec.ssl_type!r}")


def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_top_n_layers(ssl: nn.Module, top_n: int):
    """Freeze encoder, then unfreeze last top_n transformer layers and layer norms."""
    layers = getattr(ssl, "encoder_layers", [])
    if not layers:
        print("[WARN] Could not find encoder layers; leaving encoder frozen.")
        return

    top_n = min(int(top_n), len(layers))
    if top_n <= 0:
        return

    freeze_all(ssl)

    for layer in layers[-top_n:]:
        for p in layer.parameters():
            p.requires_grad = True

    for name, mod in ssl.named_modules():
        if "layer_norm" in name.lower() or name.lower().endswith("layernorm"):
            for p in mod.parameters():
                p.requires_grad = True
