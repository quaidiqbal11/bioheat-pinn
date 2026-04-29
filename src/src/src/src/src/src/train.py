"""
train.py — Training entry point for Bioheat-PINN.

Usage:
    python src/train.py --train_dir data/training --val_dir data/validation
    python src/train.py --train_dir data/training --val_dir data/validation \\
                        --epochs 140 --device cuda --checkpoint_dir checkpoints
"""
import argparse
import os
import random
import sys
import time

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from dataset import AblationVoxelDataset
from losses import (
    bioheat_residual,
    cosine_lr,
    death_loss,
    dynamic_weights,
    rmse_percent,
    zero_flux_bc_loss,
)
from model import HybridContextPINN
from utils import (
    EMA,
    ManualAdamW,
    build_feature_pyramid,
    build_point_input,
    compute_input_stats,
    predict_death_volume,
    predict_death_volume_tta,
    save_checkpoint,
    sample_boundary,
    sample_importance,
    sample_uniform,
    set_seed,
    stats_to_cpu,
    stats_to_device,
)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_resolution(cfg: CFG, resolution: str):
    set_seed(cfg.seed)

    train_root  = cfg.train_dir if cfg.train_dir else cfg.root
    train_split = None          if cfg.train_dir else cfg.split_train
    val_root    = cfg.val_dir   if cfg.val_dir   else cfg.root
    val_split   = None          if cfg.val_dir   else cfg.split_val

    train_ds = AblationVoxelDataset(
        train_root, train_split, resolution, cfg.channel_tags
    )
    val_ds = AblationVoxelDataset(
        val_root, val_split, resolution, cfg.channel_tags
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_volumes,
        shuffle=True,
        num_workers=0,
        pin_memory=cfg.device.startswith("cuda"),
    )

    input_stats_cpu = compute_input_stats(train_ds, cfg)
    input_stats     = stats_to_device(input_stats_cpu, cfg.device)

    in_dim = (
        4 * (1 + 2 * cfg.n_fourier_freqs)
        + 5
        + 5 * len(cfg.context_pool_ks)
    )
    model = HybridContextPINN(
        in_dim, cfg.model_width, cfg.model_depth
    ).to(cfg.device)
    opt = ManualAdamW(
        model.parameters(),
        lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2,
        eps=cfg.eps, weight_decay=cfg.weight_decay,
    )
    ema = EMA(model, decay=cfg.ema_decay)

    print(f"\n{'='*20} RESOLUTION {resolution} {'='*20}")
    print(f"Device : {cfg.device}  |  Train : {len(train_ds)}  |  Val : {len(val_ds)}")

    # Fixed validation subset for consistent model selection
    val_indices = list(range(len(val_ds)))
    random.shuffle(val_indices)
    val_indices = val_indices[: min(cfg.val_subset, len(val_ds))]

    best_score = best_rmse = best_std = float("inf")
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.lr = cosine_lr(ep, cfg)
        w_data, w_pde, w_ic, w_bc = dynamic_weights(ep, cfg)
        running = []
        t_ep    = time.time()

        for vol_batch in train_loader:
            vol = vol_batch[0].to(cfg.device)

            # Feature pyramid built ONCE per volume — reused for all loss terms
            pyramid = build_feature_pyramid(vol[:5], cfg)

            # ── Data loss ─────────────────────────────────────
            xyz_d, params_d, death_gt = sample_importance(
                vol, cfg, cfg.n_data_pts
            )
            t_d   = torch.full(
                (xyz_d.shape[0], 1), cfg.t_final, device=cfg.device
            )
            inp_d = build_point_input(
                xyz_d, t_d, params_d, pyramid, input_stats, cfg
            )
            _, d_pred, d_logits = model.forward_with_logits(inp_d)
            loss_data = death_loss(
                d_logits, d_pred, death_gt.unsqueeze(1), cfg
            )

            # ── PDE residual loss ─────────────────────────────
            xyz_c, params_c, _ = sample_uniform(vol, cfg, cfg.n_collocation)
            t_c = (
                torch.rand((xyz_c.shape[0], 1), device=cfg.device) * cfg.t_final
            )
            xyz_c.requires_grad_(True)
            t_c.requires_grad_(True)
            inp_c    = build_point_input(
                xyz_c, t_c, params_c, pyramid, input_stats, cfg
            )
            t_c_pred, _ = model(inp_c)
            res      = bioheat_residual(
                t_c_pred, xyz_c, t_c, params_c, model, cfg
            )
            loss_pde = res.pow(2).mean()

            # ── Initial condition loss ────────────────────────
            xyz0, params0, _ = sample_uniform(vol, cfg, cfg.n_ic_pts)
            t0   = torch.zeros((xyz0.shape[0], 1), device=cfg.device)
            inp0 = build_point_input(
                xyz0, t0, params0, pyramid, input_stats, cfg
            )
            t0p, d0p = model(inp0)
            loss_ic  = F.mse_loss(
                t0p, torch.full_like(t0p, cfg.Tb)
            ) + F.mse_loss(d0p, torch.zeros_like(d0p))

            # ── Boundary condition loss ───────────────────────
            xyz_b, params_b, normals = sample_boundary(vol, cfg, cfg.n_bc_pts)
            t_b = (
                torch.rand((xyz_b.shape[0], 1), device=cfg.device) * cfg.t_final
            )
            xyz_b.requires_grad_(True)
            inp_b   = build_point_input(
                xyz_b, t_b, params_b, pyramid, input_stats, cfg
            )
            tbnd, _ = model(inp_b)
            loss_bc = zero_flux_bc_loss(tbnd, xyz_b, normals)

            loss = (
                w_data * loss_data
                + w_pde  * loss_pde
                + w_ic   * loss_ic
                + w_bc   * loss_bc
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)

            running.append([
                loss.item(), loss_data.item(), loss_pde.item(),
                loss_ic.item(), loss_bc.item(), opt.lr,
                w_data, w_pde, w_ic, w_bc,
            ])

        arr = np.mean(running, axis=0)
        print(
            f"Ep {ep:03d} | total={arr[0]:.4e}  data={arr[1]:.4e}  "
            f"pde={arr[2]:.4e}  ic={arr[3]:.4e}  bc={arr[4]:.4e} | "
            f"lr={arr[5]:.2e} | "
            f"w=({arr[6]:.2f},{arr[7]:.2f},{arr[8]:.2f},{arr[9]:.2f}) | "
            f"{time.time()-t_ep:.1f}s"
        )

        # ── Validation with EMA weights ───────────────────────
        model.eval()
        ema.apply_shadow(model)
        rmses = []
        with torch.no_grad():
            for vi in val_indices:
                vvol   = val_ds[vi].to(cfg.device)
                target = (
                    vvol[cfg.channel_tags.index("death")].cpu().numpy()
                )
                if cfg.use_tta_in_validation:
                    pred = predict_death_volume_tta(
                        model, vvol, cfg, input_stats_cpu,
                        t_final=cfg.t_final, chunk=120000,
                        flip_axes=cfg.tta_flip_axes,
                    )
                else:
                    pred = predict_death_volume(
                        model, vvol, cfg, input_stats_cpu,
                        t_final=cfg.t_final, chunk=120000,
                    )
                rmses.append(rmse_percent(pred, target))

        val_rmse  = float(np.mean(rmses))
        val_std   = float(np.std(rmses))
        val_score = val_rmse + cfg.val_score_std_weight * val_std
        print(
            f"   Val RMSE% (n={len(val_indices)}): "
            f"{val_rmse:.3f} ± {val_std:.3f}  score={val_score:.3f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_rmse  = val_rmse
            best_std   = val_std
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            save_checkpoint(
                cfg.checkpoint_dir, resolution, model,
                input_stats_cpu, best_score, best_rmse, best_std,
            )
            print("   ⭐ New best model saved.")

        ema.restore(model)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, best_rmse, best_std, stats_to_cpu(input_stats_cpu)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Bioheat-PINN")
    p.add_argument("--train_dir",       type=str,  default="data/training")
    p.add_argument("--val_dir",         type=str,  default="data/validation")
    p.add_argument("--checkpoint_dir",  type=str,  default="checkpoints")
    p.add_argument("--epochs",          type=int,  default=140)
    p.add_argument("--device",          type=str,  default=None)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--resolutions",     nargs="+", default=["2mm", "3mm", "4mm"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    cfg = CFG(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        seed=args.seed,
        resolutions=tuple(args.resolutions),
    )
    if args.device:
        cfg.device = args.device

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    results = {}
    for res in cfg.resolutions:
        model, rmse, std, _ = train_one_resolution(cfg, res)
        results[res] = (rmse, std)

    print("\n==================== TRAINING SUMMARY ====================")
    for r, (rmse, std) in results.items():
        print(f"  {r}: best val RMSE% = {rmse:.3f} ± {std:.3f}")

