#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a saved AnimeScore RankNet checkpoint.

PRIMARY metrics (pairwise, no scalar GT required):
  pair_nll  RankNet NLL
  pair_acc  Pairwise accuracy
  pair_auc  ROC-AUC using score margin s_a - s_b

AUX metrics (utterance-level, only if --utt_csv is provided):
  MSE / LCC / SRCC / KTAU  vs. anchored MOS scores

Usage:
  python eval_ranknet_ckpt_v2.py \\
    --ckpt           ckpt/animescore_hubert_best.pt \\
    --val_pair_csv   data/pairs/pair_test_metadata.csv \\
    --wav_root       dataset/ \\
    --ssl_type       hubert \\
    --utt_csv        data/utterance_set/test_metadata.csv \\
    --utt_wav_col    shuffled_file \\
    --utt_score_col  utmos \\
    --save_pred_csv  predictions.csv
"""

import argparse
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pairwise_dataset import PairwiseMosDataset, pairwise_collate
from animescore.ssl_encoder import SSLSpec, build_ssl, freeze_all, DEFAULT_SSL_NAMES
from animescore.ranknet_model import RankNetMos
from animescore.metrics import mse, lcc, srcc, ktau, load_wav


# ── Label helpers ─────────────────────────────────────────────────────────────

def to_y_sign_and_y01(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = y.float()
    if torch.min(y) >= 0.0 and torch.max(y) <= 1.0:
        y01 = y
        y_sign = y01 * 2.0 - 1.0
    else:
        y_sign = torch.where(y > 0, torch.ones_like(y), -torch.ones_like(y))
        y01 = (y_sign > 0).float()
    return y_sign, y01


def ranknet_nll(sa: torch.Tensor, sb: torch.Tensor, y_sign: torch.Tensor) -> torch.Tensor:
    return F.softplus(-y_sign * (sa - sb)).mean()


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_ckpt_strict(model: RankNetMos, ckpt_path: str, map_location="cpu"):
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model_keys = set(model.state_dict().keys())
    matched = len(set(sd.keys()) & model_keys)
    print(f"[CKPT] keys — ckpt: {len(sd)}, model: {len(model_keys)}, matched: {matched}")
    if missing:
        print(f"[CKPT] missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"[CKPT] unexpected (first 20): {unexpected[:20]}")
    if matched < int(0.9 * len(model_keys)):
        raise RuntimeError("Too few keys matched. Check ssl_type/ssl_name match training config.")


# ── Pairwise evaluation (PRIMARY) ─────────────────────────────────────────────

@torch.no_grad()
def validate_pairwise_primary(model: RankNetMos, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    nlls, accs, deltas, y01_all = [], [], [], []

    for wav_a, wav_b, y in tqdm(loader, desc="Pairwise eval", leave=False):
        wav_a, wav_b, y = wav_a.to(device), wav_b.to(device), y.to(device)
        y_sign, y01 = to_y_sign_and_y01(y)
        sa, sb = model.score(wav_a), model.score(wav_b)
        delta = sa - sb
        nlls.append(ranknet_nll(sa, sb, y_sign).item())
        pred_sign = torch.where(delta > 0, torch.ones_like(delta), -torch.ones_like(delta))
        accs.append((pred_sign == y_sign).float().mean().item())
        deltas.append(delta.detach().cpu().numpy())
        y01_all.append(y01.detach().cpu().numpy())

    deltas_np = np.concatenate(deltas)
    y01_np    = np.concatenate(y01_all)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y01_np, deltas_np))
    except Exception:
        pos, neg = deltas_np[y01_np > 0.5], deltas_np[y01_np <= 0.5]
        auc = float("nan") if (len(pos) == 0 or len(neg) == 0) else \
              float(st.mannwhitneyu(pos, neg, alternative="two-sided")[0] / (len(pos) * len(neg)))

    return {
        "pair_nll":    float(np.mean(nlls)),
        "pair_acc":    float(np.mean(accs)),
        "pair_auc":    auc,
        "margin_mean": float(np.mean(deltas_np)),
        "margin_std":  float(np.std(deltas_np)),
        "n_pairs":     float(len(deltas_np)),
    }


# ── Utterance-level evaluation (AUX) ─────────────────────────────────────────

@torch.no_grad()
def predict_utterances(model, utt_csv, wav_col, score_col, target_sr, device, batch_size=8) -> pd.DataFrame:
    df = pd.read_csv(utt_csv)[[wav_col, score_col]].dropna()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col]).reset_index(drop=True)
    paths  = df[wav_col].astype(str).tolist()
    y_true = df[score_col].to_numpy(dtype=np.float64)
    y_pred = []
    model.eval()
    for i in tqdm(range(0, len(paths), batch_size), desc="Utterance preds"):
        chunk = paths[i:i+batch_size]
        wavs  = [load_wav(p, target_sr, device) for p in chunk]
        maxlen = max(w.numel() for w in wavs)
        batch  = torch.stack([F.pad(w, (0, maxlen - w.numel())) for w in wavs])
        y_pred.append(model.score(batch).detach().cpu().numpy().astype(np.float64))
    return pd.DataFrame({"wav": paths, "aux_score": y_true,
                         "pred_score": np.concatenate(y_pred)})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--val_pair_csv",   required=True)
    ap.add_argument("--wav_root",       required=True)
    ap.add_argument("--batch_size",     type=int, default=8)
    ap.add_argument("--num_workers",    type=int, default=2)
    ap.add_argument("--ssl_type",       default="wav2vec2",
                    choices=["wav2vec2", "wavlm", "hubert", "data2vec", "whisper-enc"])
    ap.add_argument("--ssl_name",       default="")
    ap.add_argument("--utt_csv",        default="")
    ap.add_argument("--utt_wav_col",    default="audio")
    ap.add_argument("--utt_score_col",  default="score")
    ap.add_argument("--utt_batch_size", type=int, default=8)
    ap.add_argument("--save_pred_csv",  default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device   = torch.device(args.device)
    ssl_name = args.ssl_name.strip() or DEFAULT_SSL_NAMES[args.ssl_type]
    ssl      = build_ssl(SSLSpec(args.ssl_type, ssl_name, target_sr=16000, feat_dim=0)).to(device)
    target_sr = getattr(ssl, "target_sr", 16000)
    print(f"[INFO] SSL: type={args.ssl_type} name={ssl_name} feat_dim={ssl.feat_dim}")

    freeze_all(ssl)
    model = RankNetMos(ssl=ssl).to(device)
    print("[INFO] Loading checkpoint...")
    load_ckpt_strict(model, args.ckpt, map_location="cpu")
    model.eval()

    val_ds = PairwiseMosDataset(args.val_pair_csv, args.wav_root, max_sec=None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=pairwise_collate, pin_memory=True)

    print("\n=== Pairwise Evaluation (PRIMARY) ===")
    out = validate_pairwise_primary(model, val_loader, device)
    print(f"[PAIRWISE] N={int(out['n_pairs'])} | NLL={out['pair_nll']:.6f} | "
          f"Acc={out['pair_acc']:.6f} | AUC={out['pair_auc']:.6f}")
    print(f"[PAIRWISE] margin mean={out['margin_mean']:+.4f} std={out['margin_std']:.4f}")

    if args.utt_csv.strip():
        print("\n=== Utterance-level Evaluation (AUX) ===")
        pred_df = predict_utterances(model, args.utt_csv, args.utt_wav_col, args.utt_score_col,
                                     target_sr, device, args.utt_batch_size)
        yt, yp = pred_df["aux_score"].to_numpy(), pred_df["pred_score"].to_numpy()
        print(f"[AUX-UTT] N={len(pred_df)} | MSE={mse(yt,yp):.6f} | "
              f"LCC={lcc(yt,yp):.6f} | SRCC={srcc(yt,yp):.6f} | KTAU={ktau(yt,yp):.6f}")
        if args.save_pred_csv.strip():
            os.makedirs(os.path.dirname(args.save_pred_csv) or ".", exist_ok=True)
            pred_df.to_csv(args.save_pred_csv, index=False)
            print(f"[AUX-UTT] Saved: {args.save_pred_csv}")

    print("\n=== Done ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
