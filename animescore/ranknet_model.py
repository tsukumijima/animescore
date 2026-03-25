"""
RankNet MOS model: SSL encoder + BiLSTM + mean pooling + MLP.
Shared by training and evaluation scripts.
"""

import torch
import torch.nn as nn


class RankNetMos(nn.Module):
    """
    SSL → BiLSTM → mean pool → MLP → scalar score s(x).

    Pairwise ranking: P(a ≻ b) = σ(s_a − s_b).
    """

    def __init__(
        self,
        ssl: nn.Module,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ssl = ssl

        feat_dim = getattr(self.ssl, "feat_dim", None)
        if feat_dim is None:
            raise RuntimeError("ssl wrapper must expose feat_dim")

        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = 2 * lstm_hidden

        self.mlp = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def extract_feats(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.ssl(wav)
        return out["last_hidden_state"]

    def score(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feats(wav)       # [B, T', D]
        z, _ = self.bilstm(feats)             # [B, T', 2H]
        zbar = z.mean(dim=1)                  # [B, 2H]
        return self.mlp(zbar).squeeze(-1)     # [B]
