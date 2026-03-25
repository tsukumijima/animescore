#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train AnimeMOS RankNet with a configurable SSL encoder.

Pairwise training (RankNet logistic loss) + optional utterance-level evaluation.

Supported SSL backbones:
  wav2vec2    facebook/wav2vec2-base-960h (default)
  hubert      facebook/hubert-base-ls960
  wavlm       microsoft/wavlm-base
  data2vec    facebook/data2vec-audio-base-960h
  whisper-enc openai/whisper-small (encoder only)

Usage:
  # Frozen encoder (recommended)
  python train_ranknet_v2.py \\
    --train_pair_csv data/pairs/pair_train_metadata.csv \\
    --val_pair_csv   data/pairs/pair_test_metadata.csv \\
    --wav_root       dataset/ \\
    --utt_csv        data/utterance_set/test_metadata.csv \\
    --ssl_type       hubert --freeze_ssl \\
    --save_path      ckpt/animescore_hubert.pt

  # Unfreeze top-2 encoder layers
  python train_ranknet_v2.py ... --ssl_type wavlm --unfreeze_top_n 2
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pairwise_dataset import PairwiseMosDataset, pairwise_collate
from animescore.ssl_encoder import SSLSpec, build_ssl, freeze_all, unfreeze_top_n_layers, DEFAULT_SSL_NAMES
from animescore.ranknet_model import RankNetMos
from animescore.metrics import mse, lcc, srcc, ktau, load_wav


# ── RankNet loss ──────────────────────────────────────────────────────────────

def ranknet_loss(sa: torch.Tensor, sb: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    """y01 in {0,1}: 1 means a preferred over b."""
    y = y01 * 2 - 1  # -> {-1, +1}
    return torch.nn.functional.softplus(-y * (sa - sb)).mean()


# ── Partial correlation / controlled regression (proxy analysis) ──────────────

def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> Tuple[float, float, int]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    fn = st.pearsonr if method == "pearson" else st.spearmanr
    r, p = fn(x[m], y[m])
    return float(r), float(p), n


def partial_corr(x, y, controls, method="pearson"):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    C = np.asarray(controls, np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    n = int(m.sum())
    if n < 20:
        return float("nan"), float("nan"), n
    Xc = np.column_stack([np.ones(n), C[m]])
    bx, *_ = np.linalg.lstsq(Xc, x[m], rcond=None)
    by, *_ = np.linalg.lstsq(Xc, y[m], rcond=None)
    rx, ry = x[m] - Xc @ bx, y[m] - Xc @ by
    fn = st.spearmanr if method == "spearman" else st.pearsonr
    r, p = fn(rx, ry)
    return float(r), float(p), n


def controlled_regression(y, X, feature_names):
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(X, has_constant="add")
        m = sm.OLS(y, Xc).fit(cov_type="HC3")
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        yz = (y - y.mean()) / (y.std() + 1e-12)
        mz = sm.OLS(yz, sm.add_constant(Xz, has_constant="add")).fit(cov_type="HC3")
        rows = [{"name": "const", "coef": float(m.params[0]), "beta": float(mz.params[0]), "p": float(m.pvalues[0])}]
        for i, fn in enumerate(feature_names):
            rows.append({"name": fn, "coef": float(m.params[i+1]), "beta": float(mz.params[i+1]), "p": float(m.pvalues[i+1])})
        return {"r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "rows": rows, "summary": str(m.summary())}
    except Exception as e:
        Xc = np.column_stack([np.ones(len(y)), X])
        b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        yhat = Xc @ b
        r2 = 1.0 - float(np.sum((y - yhat)**2)) / (float(np.sum((y - y.mean())**2)) + 1e-12)
        rows = [{"name": "const", "coef": float(b[0]), "beta": float("nan"), "p": float("nan")}]
        for i, fn in enumerate(feature_names):
            rows.append({"name": fn, "coef": float(b[i+1]), "beta": float("nan"), "p": float("nan")})
        return {"r2": r2, "adj_r2": float("nan"), "rows": rows, "summary": f"Fallback numpy OLS. statsmodels error: {e}"}


# ── Validation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def validate_pairwise(model: RankNetMos, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses, accs = [], []
    for wav_a, wav_b, y in loader:
        wav_a, wav_b, y = wav_a.to(device), wav_b.to(device), y.to(device)
        sa, sb = model.score(wav_a), model.score(wav_b)
        losses.append(ranknet_loss(sa, sb, y).item())
        accs.append(((sa > sb).float() == y).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def eval_utterance_metrics(model, utt_csv, wav_col, score_col, target_sr, device, batch_size=8):
    df = pd.read_csv(utt_csv)[[wav_col, score_col]].dropna()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col]).reset_index(drop=True)
    y_true = df[score_col].to_numpy(dtype=np.float64)
    paths = df[wav_col].astype(str).tolist()
    y_pred = []
    model.eval()
    for i in tqdm(range(0, len(paths), batch_size), desc="Utterance eval", leave=False):
        chunk = paths[i:i+batch_size]
        wavs = [load_wav(p, target_sr, device) for p in chunk]
        maxlen = max(w.numel() for w in wavs)
        batch = torch.stack([torch.nn.functional.pad(w, (0, maxlen - w.numel())) for w in wavs])
        y_pred.append(model.score(batch).detach().cpu().numpy())
    y_pred = np.concatenate(y_pred).astype(np.float64)
    return {"MSE": mse(y_true, y_pred), "LCC": lcc(y_true, y_pred),
            "SRCC": srcc(y_true, y_pred), "KTAU": ktau(y_true, y_pred), "N": len(y_true)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_pair_csv", required=True)
    ap.add_argument("--val_pair_csv",   required=True)
    ap.add_argument("--wav_root",       required=True)
    ap.add_argument("--utt_csv",        default="", help="CSV with wav_col + score_col for utterance metrics")
    ap.add_argument("--utt_wav_col",    default="audio")
    ap.add_argument("--utt_score_col",  default="score")
    ap.add_argument("--utt_batch_size", type=int, default=8)
    ap.add_argument("--max_sec",        type=float, default=0.0)

    # SSL
    ap.add_argument("--ssl_type",  default="wav2vec2",
                    choices=["wav2vec2", "wavlm", "hubert", "data2vec", "whisper-enc"])
    ap.add_argument("--ssl_name",  default="", help="HuggingFace model ID (uses default if empty)")
    ap.add_argument("--freeze_ssl",      action="store_true")
    ap.add_argument("--unfreeze_top_n",  type=int, default=0)

    # Training
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch_size",  type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=3e-5)
    ap.add_argument("--num_workers", type=int,   default=2)
    ap.add_argument("--save_path",   default="mos_ranknet.pt")

    # Proxy analysis (optional)
    ap.add_argument("--proxy_csv",      default="")
    ap.add_argument("--proxy_wav_col",  default="wav")
    ap.add_argument("--proxy_controls", default="")
    ap.add_argument("--proxy_out_json", default="")

    # Misc
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",   type=int, default=0)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ssl_name = args.ssl_name.strip() or DEFAULT_SSL_NAMES[args.ssl_type]
    ssl = build_ssl(SSLSpec(args.ssl_type, ssl_name, target_sr=16000, feat_dim=0)).to(device)
    target_sr = getattr(ssl, "target_sr", 16000)
    print(f"[INFO] SSL: type={args.ssl_type} name={ssl_name} target_sr={target_sr} feat_dim={ssl.feat_dim}")

    if args.unfreeze_top_n > 0:
        unfreeze_top_n_layers(ssl, args.unfreeze_top_n)
        print(f"[INFO] SSL partially trainable: top-{args.unfreeze_top_n} layers unfrozen")
    else:
        freeze_all(ssl)
        print("[INFO] SSL frozen")

    model = RankNetMos(ssl=ssl).to(device)

    max_sec = args.max_sec if args.max_sec > 0 else None
    train_ds = PairwiseMosDataset(args.train_pair_csv, args.wav_root, max_sec=max_sec)
    val_ds   = PairwiseMosDataset(args.val_pair_csv,   args.wav_root, max_sec=max_sec)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=pairwise_collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=pairwise_collate, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    best_val_loss, best_epoch = float("inf"), -1
    best_path = args.save_path.replace(".pt", "_best.pt")

    print("[INFO] Starting pairwise RankNet training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        for wav_a, wav_b, y in train_loader:
            wav_a, wav_b, y = wav_a.to(device), wav_b.to(device), y.to(device)
            loss = ranknet_loss(model.score(wav_a), model.score(wav_b), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        val_loss, val_acc = validate_pairwise(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] train={total_loss/max(1,n_steps):.6f} "
              f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] epoch={epoch:02d} → {best_path}")

    torch.save(model.state_dict(), args.save_path)
    print(f"[INFO] Final checkpoint: {args.save_path}")
    print(f"[INFO] Best epoch={best_epoch}, val_loss={best_val_loss:.6f}")

    # Extended evaluation
    print("\n=== Extended Evaluation ===")
    val_loss, val_acc = validate_pairwise(model, val_loader, device)
    print(f"[PAIRWISE] val_loss={val_loss:.6f} | val_acc={val_acc:.6f}")

    if args.utt_csv.strip():
        utt_m = eval_utterance_metrics(model, args.utt_csv, args.utt_wav_col,
                                       args.utt_score_col, target_sr, device, args.utt_batch_size)
        print(f"[UTTERANCE] MSE={utt_m['MSE']:.6f} LCC={utt_m['LCC']:.6f} "
              f"SRCC={utt_m['SRCC']:.6f} KTAU={utt_m['KTAU']:.6f} N={utt_m['N']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
