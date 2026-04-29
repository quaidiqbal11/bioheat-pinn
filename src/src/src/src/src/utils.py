"""
utils.py — Feature building, sampling utilities, optimiser, EMA, and checkpointing.
"""
import math
import os
import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from config import CFG
    from model import HybridContextPINN


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# Input normalisation statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_input_stats(
    dataset: Dataset, cfg: "CFG"
) -> Dict[str, torch.Tensor]:
    idxs = list(range(len(dataset)))
    rng  = random.Random(cfg.seed)
    rng.shuffle(idxs)
    idxs = idxs[: min(cfg.stats_max_volumes, len(dataset))]

    s  = torch.zeros(5, dtype=torch.float64)
    s2 = torch.zeros(5, dtype=torch.float64)
    n  = 0
    for idx in idxs:
        vol    = dataset[idx].float()
        params = vol[:5].reshape(5, -1).double()
        s  += params.sum(dim=1)
        s2 += (params ** 2).sum(dim=1)
        n  += params.shape[1]

    mean = (s / max(n, 1)).float()
    var  = (s2 / max(n, 1) - mean.double() ** 2).float().clamp_min(1e-8)
    return {"mean": mean, "std": var.sqrt()}


def stats_to_device(
    stats: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in stats.items()}


def stats_to_cpu(
    stats: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in stats.items()}


def normalize_params(
    params_raw: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    m = stats["mean"].to(params_raw.device).view(1, -1)
    s = stats["std"].to(params_raw.device).view(1, -1)
    return (params_raw - m) / s


# ─────────────────────────────────────────────────────────────────────────────
# Feature building
# ─────────────────────────────────────────────────────────────────────────────
def fourier_features(x: torch.Tensor, n_freqs: int) -> torch.Tensor:
    outs = [x]
    for k in range(n_freqs):
        f = (2.0 ** k) * math.pi
        outs += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(outs, dim=1)


def build_feature_pyramid(
    vol_params_5: torch.Tensor, cfg: "CFG"
) -> List[torch.Tensor]:
    """Build multi-scale avg-pool pyramid once per volume — reuse for all loss terms."""
    x     = vol_params_5.unsqueeze(0)   # (1, 5, D, H, W)
    feats = []
    for k in cfg.context_pool_ks:
        if k == 1:
            feats.append(x)
        else:
            feats.append(
                F.avg_pool3d(x, kernel_size=k, stride=1, padding=k // 2)
            )
    return feats


def sample_volume_features(
    feature_vol: torch.Tensor, xyz: torch.Tensor
) -> torch.Tensor:
    grid = xyz.detach().view(1, -1, 1, 1, 3) * 2.0 - 1.0
    with torch.no_grad():
        sampled = F.grid_sample(
            feature_vol, grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
    return sampled.view(feature_vol.shape[1], -1).transpose(0, 1)


def sample_context_features(
    pyramid: List[torch.Tensor],
    xyz: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    pieces = []
    for feat in pyramid:
        pieces.append(
            normalize_params(sample_volume_features(feat, xyz), stats)
        )
    return torch.cat(pieces, dim=1)


def build_point_input(
    xyz:     torch.Tensor,
    t:       torch.Tensor,
    params_raw: torch.Tensor,
    pyramid: List[torch.Tensor],
    stats:   Dict[str, torch.Tensor],
    cfg:     "CFG",
) -> torch.Tensor:
    coord_time     = torch.cat([xyz, t], dim=1)
    coord_time_enc = fourier_features(2.0 * coord_time - 1.0, cfg.n_fourier_freqs)
    params_norm    = normalize_params(params_raw, stats)
    context        = sample_context_features(pyramid, xyz, stats)
    return torch.cat([coord_time_enc, params_norm, context], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────
def _gather(vol, cfg, zz, yy, xx):
    idx   = {t: i for i, t in enumerate(cfg.channel_tags)}
    appl  = vol[idx["applicator"],    zz, yy, xx]
    perf  = vol[idx["perfusion"],     zz, yy, xx]
    rho   = vol[idx["density"],       zz, yy, xx]
    c     = vol[idx["specific_heat"], zz, yy, xx]
    slc   = vol[idx["slice"],         zz, yy, xx]
    death = vol[idx["death"],         zz, yy, xx]
    return torch.stack([appl, perf, rho, c, slc], dim=1), death


def _coords(xx, yy, zz, w, h, d):
    x = xx.float() / max(w - 1, 1)
    y = yy.float() / max(h - 1, 1)
    z = zz.float() / max(d - 1, 1)
    return torch.stack([x, y, z], dim=1)


def sample_uniform(vol: torch.Tensor, cfg: "CFG", n_pts: int):
    device = vol.device
    _, d, h, w = vol.shape
    zz = torch.randint(0, d, (n_pts,), device=device)
    yy = torch.randint(0, h, (n_pts,), device=device)
    xx = torch.randint(0, w, (n_pts,), device=device)
    params, death = _gather(vol, cfg, zz, yy, xx)
    return _coords(xx, yy, zz, w, h, d), params, death


def sample_importance(vol: torch.Tensor, cfg: "CFG", n_pts: int):
    device = vol.device
    _, d, h, w = vol.shape
    idx     = {t: i for i, t in enumerate(cfg.channel_tags)}
    dv      = vol[idx["death"]].reshape(-1)
    av      = vol[idx["applicator"]].reshape(-1)
    bv      = 4.0 * dv * (1.0 - dv)
    weights = (
        1e-6
        + cfg.sample_w_death      * dv
        + cfg.sample_w_boundary   * bv
        + cfg.sample_w_applicator * av
    )
    weights = weights / weights.sum()
    n_focus = int(round(cfg.focus_frac * n_pts))
    n_uni   = n_pts - n_focus
    focus   = torch.multinomial(weights, n_focus, replacement=True)
    uni     = torch.randint(0, dv.numel(), (n_uni,), device=device)
    all_idx = torch.cat([focus, uni])
    zz = all_idx // (h * w)
    rem = all_idx % (h * w)
    yy = rem // w
    xx = rem % w
    params, death = _gather(vol, cfg, zz, yy, xx)
    return _coords(xx, yy, zz, w, h, d), params, death


def sample_boundary(vol: torch.Tensor, cfg: "CFG", n_pts: int):
    device = vol.device
    _, d, h, w = vol.shape
    face_id = torch.randint(0, 6, (n_pts,), device=device)
    zz = torch.randint(0, d, (n_pts,), device=device)
    yy = torch.randint(0, h, (n_pts,), device=device)
    xx = torch.randint(0, w, (n_pts,), device=device)
    normals = torch.zeros((n_pts, 3), device=device)
    for fi, (dim_arr, dim_val, ni, nv) in enumerate([
        (xx, 0,     0, -1.0), (xx, w - 1, 0,  1.0),
        (yy, 0,     1, -1.0), (yy, h - 1, 1,  1.0),
        (zz, 0,     2, -1.0), (zz, d - 1, 2,  1.0),
    ]):
        m = face_id == fi
        dim_arr[m] = dim_val
        normals[m, ni] = nv
    params, _ = _gather(vol, cfg, zz, yy, xx)
    return _coords(xx, yy, zz, w, h, d), params, normals


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_death_volume(
    model:       "HybridContextPINN",
    vol:         torch.Tensor,
    cfg:         "CFG",
    input_stats: Dict[str, torch.Tensor],
    t_final:     float,
    chunk:       int = 120000,
) -> np.ndarray:
    model.eval()
    device = cfg.device
    vol    = vol.to(device)
    _, d, h, w = vol.shape
    idx    = {t: i for i, t in enumerate(cfg.channel_tags)}

    pyramid   = build_feature_pyramid(vol[:5], cfg)
    stats_dev = stats_to_device(input_stats, device)

    appl = vol[idx["applicator"]].reshape(-1)
    perf = vol[idx["perfusion"]].reshape(-1)
    rho  = vol[idx["density"]].reshape(-1)
    c    = vol[idx["specific_heat"]].reshape(-1)
    slc  = vol[idx["slice"]].reshape(-1)

    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    x  = xx.reshape(-1).float() / max(w - 1, 1)
    y  = yy.reshape(-1).float() / max(h - 1, 1)
    z  = zz.reshape(-1).float() / max(d - 1, 1)
    t  = torch.full_like(x, float(t_final))
    pr = torch.stack([appl, perf, rho, c, slc], dim=1)

    out = torch.empty((d * h * w,), device=device)
    for s in range(0, out.numel(), chunk):
        e   = min(out.numel(), s + chunk)
        inp = build_point_input(
            torch.stack([x[s:e], y[s:e], z[s:e]], dim=1),
            t[s:e, None], pr[s:e], pyramid, stats_dev, cfg,
        )
        _, dp = model(inp)
        out[s:e] = dp.squeeze(1)

    return out.view(d, h, w).cpu().numpy()


@torch.no_grad()
def predict_death_volume_tta(
    model:       "HybridContextPINN",
    vol:         torch.Tensor,
    cfg:         "CFG",
    input_stats: Dict[str, torch.Tensor],
    t_final:     float,
    chunk:       int,
    flip_axes:   Tuple[Tuple[int, ...], ...],
) -> np.ndarray:
    """Test-time augmentation: average predictions over axis-aligned flips."""
    preds = []
    for axes in flip_axes:
        v = torch.flip(vol, dims=list(axes)) if axes else vol
        p = predict_death_volume(
            model, v, cfg, input_stats, t_final=t_final, chunk=chunk
        )
        if axes:
            p = np.flip(p, axis=tuple(a - 1 for a in axes))
        preds.append(p.copy())
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser + EMA
# ─────────────────────────────────────────────────────────────────────────────
class ManualAdamW:
    """AdamW implemented without torch.optim (avoids optimizer state issues with EMA)."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr; self.b1 = beta1; self.b2 = beta2
        self.eps = eps; self.wd = weight_decay; self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd > 0:
                p.mul_(1.0 - self.lr * self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            mhat = self.m[i] / (1.0 - self.b1 ** self.t)
            vhat = self.v[i] / (1.0 - self.b2 ** self.t)
            p.addcdiv_(mhat, vhat.sqrt().add_(self.eps), value=-self.lr)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class EMA:
    """Exponential Moving Average of model weights for stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(
                sd[k].detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    checkpoint_dir: str,
    resolution:     str,
    model:          nn.Module,
    input_stats:    Dict[str, torch.Tensor],
    best_score:     float,
    best_rmse:      float,
    best_std:       float,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"bioheat_pinn_{resolution}_best.pt")
    torch.save(
        {
            "state_dict":    {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
            "input_stats":   stats_to_cpu(input_stats),
            "best_score":    float(best_score),
            "best_val_rmse": float(best_rmse),
            "best_val_std":  float(best_std),
        },
        path,
    )
    print(f"   Checkpoint saved: {path}")


def load_checkpoint_model(
    checkpoint_dir: str,
    resolution:     str,
    cfg:            "CFG",
):
    from model import HybridContextPINN
    path = os.path.join(checkpoint_dir, f"bioheat_pinn_{resolution}_best.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    in_dim = (
        4 * (1 + 2 * cfg.n_fourier_freqs)
        + 5
        + 5 * len(cfg.context_pool_ks)
    )
    ckpt  = torch.load(path, map_location=cfg.device)
    model = HybridContextPINN(in_dim, cfg.model_width, cfg.model_depth).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt["input_stats"]
