"""
losses.py — Physics residual and all loss functions for Bioheat-PINN.
"""
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from model import HybridContextPINN
    from config import CFG


# ── Bioheat PDE residual ──────────────────────────────────────────────────────
def bioheat_residual(
    t_pred:      torch.Tensor,
    coords_xyz:  torch.Tensor,
    t:           torch.Tensor,
    params_raw:  torch.Tensor,
    model:       "HybridContextPINN",
    cfg:         "CFG",
) -> torch.Tensor:
    """
    Computes the normalised Pennes bioheat residual:
        rho * c * dT/dt - div(kappa * grad(T)) - omega*(Tb - T) - Q_app = 0
    All gradients are via autograd (coords_xyz and t must have requires_grad=True).
    """
    appl = params_raw[:, 0:1]
    perf = params_raw[:, 1:2]
    rho  = params_raw[:, 2:3]
    c    = params_raw[:, 3:4]
    slc  = params_raw[:, 4:5]

    kx = (cfg.k0 + cfg.k1 * torch.sigmoid(10.0 * (slc - 0.5))) * model.k_multiplier()

    # Time derivative
    t_t = torch.autograd.grad(
        t_pred, t,
        grad_outputs=torch.ones_like(t_pred),
        create_graph=True, retain_graph=True,
    )[0]

    # Spatial gradients
    grads = torch.autograd.grad(
        t_pred, coords_xyz,
        grad_outputs=torch.ones_like(t_pred),
        create_graph=True, retain_graph=True,
    )[0]
    tx, ty, tz = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]

    def div_k(dv: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            kx * dv, coords_xyz,
            grad_outputs=torch.ones_like(kx * dv),
            create_graph=True, retain_graph=True,
        )[0]

    div_kg = (
        div_k(tx)[:, 0:1]
        + div_k(ty)[:, 1:2]
        + div_k(tz)[:, 2:3]
    )

    q_ext  = cfg.q0 * appl
    q_bio  = model.perf_multiplier() * perf * (cfg.Tb - t_pred)
    res    = rho * c * t_t - div_kg - q_bio - q_ext

    # Normalise by local scale to avoid magnitude dominance
    scale  = (
        (rho * c).abs()
        + q_ext.abs()
        + (model.perf_multiplier() * perf).abs()
        + 1.0
    )
    return res / scale


# ── Zero-flux Neumann BC loss ─────────────────────────────────────────────────
def zero_flux_bc_loss(
    tbnd:       torch.Tensor,
    coords_xyz: torch.Tensor,
    normals:    torch.Tensor,
) -> torch.Tensor:
    """Penalises non-zero normal flux at domain boundaries: (grad T · n)^2."""
    grads = torch.autograd.grad(
        tbnd, coords_xyz,
        grad_outputs=torch.ones_like(tbnd),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads * normals).sum(dim=1, keepdim=True) ** 2).mean()


# ── Death-field loss ──────────────────────────────────────────────────────────
def dice_loss_soft(
    pred:    torch.Tensor,
    target:  torch.Tensor,
    eps:     float = 1e-6,
) -> torch.Tensor:
    num = 2.0 * (pred * target).sum() + eps
    den = (pred * pred).sum() + (target * target).sum() + eps
    return 1.0 - num / den


def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma:  float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt  = (
        target * torch.sigmoid(logits)
        + (1 - target) * (1 - torch.sigmoid(logits))
    )
    return ((1.0 - pt) ** gamma) * bce


def death_loss(
    logits: torch.Tensor,
    pred:   torch.Tensor,
    target: torch.Tensor,
    cfg:    "CFG",
) -> torch.Tensor:
    """
    Composite death-field loss combining:
      MSE + Huber + Focal-BCE + Soft-Dice + Top-k hard examples
    Weighted by spatial importance (ablation boundary and core).
    """
    target = target.clamp(0.0, 1.0)
    pred   = pred.clamp(1e-5, 1.0 - 1e-5)

    w = (
        (1.0 + cfg.pos_weight_strength      * target)
        * (1.0 + cfg.boundary_weight_strength * 4.0 * target * (1.0 - target))
    )
    w = w / (w.mean() + 1e-8)

    mse_m   = (pred - target) ** 2
    huber_m = F.smooth_l1_loss(pred, target, beta=0.05, reduction="none")
    bce_m   = focal_bce(logits, target, cfg.focal_gamma)

    base = w * (0.45 * mse_m + 0.20 * huber_m + 0.35 * bce_m)
    k    = max(1, int(0.25 * base.numel()))
    topk = torch.topk(base.reshape(-1), k=k, largest=True).values.mean()

    return (
        cfg.mse_weight   * (w * mse_m).mean()
        + cfg.huber_weight * (w * huber_m).mean()
        + cfg.bce_weight   * (w * bce_m).mean()
        + cfg.dice_weight  * dice_loss_soft(pred, target)
        + cfg.topk_weight  * topk
    )


# ── Evaluation metric ─────────────────────────────────────────────────────────
def rmse_percent(
    pred:   np.ndarray,
    target: np.ndarray,
    eps:    float = 1e-8,
) -> float:
    """Full-volume RMSE%, as used in all reported results."""
    return 100.0 * math.sqrt(
        np.mean((np.clip(pred, 0, 1) - target) ** 2) + eps
    )


# ── Scheduler helpers ─────────────────────────────────────────────────────────
def cosine_lr(epoch: int, cfg: "CFG") -> float:
    """Cosine annealing with linear warm-up."""
    if epoch <= cfg.warmup_epochs:
        return cfg.lr * epoch / max(cfg.warmup_epochs, 1)
    p = min(
        max(
            (epoch - cfg.warmup_epochs)
            / max(cfg.epochs - cfg.warmup_epochs, 1),
            0.0,
        ),
        1.0,
    )
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * p))


def dynamic_weights(epoch: int, cfg: "CFG"):
    """Gradually ramp PDE and BC loss weights after warm-up."""
    pde_s = min(1.0, max(epoch - 8, 0) / max(cfg.pde_warmup_epochs, 1))
    bc_s  = min(1.0, max(epoch - 4, 0) / max(cfg.bc_warmup_epochs,  1))
    return cfg.w_data, cfg.w_pde * pde_s, cfg.w_ic, cfg.w_bc * bc_s
