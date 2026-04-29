"""
test.py — Testing and evaluation entry point for Bioheat-PINN.

Usage:
    python src/test.py --test_dir data/testing --checkpoint_dir checkpoints
    python src/test.py --test_dir data/testing --checkpoint_dir checkpoints \\
                       --device cuda --save_figures --n_qualitative 4
"""
import argparse
import os
import random
import sys
import time
from typing import Dict, Optional

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG
from dataset import TestVoxelDataset
from losses import rmse_percent
from utils import (
    load_checkpoint_model,
    predict_death_volume,
    predict_death_volume_tta,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_cuda_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def savefig(fig, path_no_ext: str, dpi: int = 300):
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    fig.savefig(
        path_no_ext + ".png",
        dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white",
    )
    fig.savefig(
        path_no_ext + ".pdf",
        bbox_inches="tight", pad_inches=0.02, facecolor="white",
    )
    print("Saved:", path_no_ext + ".png")
    plt.close(fig)


def make_overlay(
    death: np.ndarray,
    vessels_like: np.ndarray,
    applicator: np.ndarray,
) -> np.ndarray:
    death        = np.clip(death, 0.0, 1.0)
    vessels_like = np.clip(vessels_like, 0.0, 1.0)
    applicator   = np.clip(applicator,   0.0, 1.0)
    rgb          = np.zeros((*death.shape[:2], 3), dtype=np.float32)
    rgb[..., 0]  = np.clip(death + applicator, 0.0, 1.0)
    rgb[..., 1]  = applicator
    rgb[..., 2]  = vessels_like
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Per-resolution evaluation
# ─────────────────────────────────────────────────────────────────────────────
def eval_resolution(
    testing_dir: str,
    resolution:  str,
    model,
    input_stats: Dict,
    cfg:         CFG,
    max_cases:   Optional[int] = None,
) -> dict:
    ds    = TestVoxelDataset(testing_dir, resolution, cfg.channel_tags)
    n     = len(ds) if max_cases is None else min(len(ds), max_cases)
    rmses, times = [], []

    for i in range(n):
        vol    = ds[i]
        target = vol[list(cfg.channel_tags).index("death")].numpy()
        safe_cuda_sync(cfg.device)
        t0 = time.time()
        if cfg.use_tta_in_testing:
            pred = predict_death_volume_tta(
                model, vol, cfg, input_stats,
                t_final=cfg.t_final, chunk=120000,
                flip_axes=cfg.tta_flip_axes,
            )
        else:
            pred = predict_death_volume(
                model, vol, cfg, input_stats,
                t_final=cfg.t_final, chunk=120000,
            )
        safe_cuda_sync(cfg.device)
        rmses.append(rmse_percent(pred, target))
        times.append(time.time() - t0)

    rmses = np.array(rmses)
    times = np.array(times)
    return {
        "n":         n,
        "rmse_mean": float(rmses.mean()),
        "rmse_std":  float(rmses.std()),
        "time_mean": float(times.mean()),
        "time_std":  float(times.std()),
        "fps_mean":  float(1.0 / max(times.mean(), 1e-9)),
        "rmse_all":  rmses,
        "time_all":  times,
        "dataset":   ds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure saving
# ─────────────────────────────────────────────────────────────────────────────
def save_summary_figures(
    results:    Dict[str, dict],
    figure_dir: str,
    resolutions,
):
    os.makedirs(figure_dir, exist_ok=True)
    ordered     = list(resolutions)
    rmse_lists  = [results[r]["rmse_all"]  for r in ordered]
    fps_vals    = [results[r]["fps_mean"]  for r in ordered]
    rmse_means  = [results[r]["rmse_mean"] for r in ordered]
    rmse_stds   = [results[r]["rmse_std"]  for r in ordered]
    time_means  = [results[r]["time_mean"] for r in ordered]
    voxel_sizes = [int(r.replace("mm", "")) for r in ordered]

    # 1. RMSE violin
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=250, facecolor="white")
    vp = ax.violinplot(rmse_lists, showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_alpha(0.35)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(ordered)
    ax.set_ylabel("RMSE (%)")
    ax.set_title("RMSE distribution vs voxel size (testing)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, os.path.join(figure_dir, "Fig_RMSE_violin"))

    # 2. FPS vs voxel size
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=250, facecolor="white")
    ax.plot(voxel_sizes, fps_vals, marker="o", linewidth=2)
    ax.set_xlabel("Voxel size (mm)")
    ax.set_ylabel("Inference speed (volumes/s)")
    ax.set_title("Inference speed vs voxel size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, os.path.join(figure_dir, "Fig_FPS_vs_voxel"))

    # 3. RMSE bar chart
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=250, facecolor="white")
    ax.bar(ordered, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.8)
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Average RMSE ± STD (testing)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, os.path.join(figure_dir, "Fig_RMSE_bar"))

    # 4. Inference time
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=250, facecolor="white")
    ax.plot(voxel_sizes, time_means, marker="s", linewidth=2)
    ax.set_xlabel("Voxel size (mm)")
    ax.set_ylabel("Inference time per volume (s)")
    ax.set_title("Inference time vs voxel size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, os.path.join(figure_dir, "Fig_time_vs_voxel"))


def show_target_vs_output(
    vol:        torch.Tensor,
    pred_death: np.ndarray,
    channel_tags,
    title_suffix: str = "",
    save_path:    Optional[str] = None,
    z_slice:      Optional[int] = None,
):
    idx      = {t: i for i, t in enumerate(channel_tags)}
    death_gt = vol[idx["death"]].numpy()
    appl     = vol[idx["applicator"]].numpy()
    slc      = vol[idx["slice"]].numpy()
    D        = death_gt.shape[0]
    z        = D // 2 if z_slice is None else z_slice

    tgt_rgb = make_overlay(death_gt[z], slc[z], appl[z])
    out_rgb = make_overlay(pred_death[z], slc[z], appl[z])

    fig = plt.figure(figsize=(8.0, 4.0), dpi=250, facecolor="white")
    gs  = fig.add_gridspec(
        1, 2, left=0.02, right=0.98, bottom=0.04, top=0.88, wspace=0.02
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for a, img, ttl in zip(
        [ax1, ax2],
        [tgt_rgb, out_rgb],
        [f"Target{title_suffix}", f"Output{title_suffix}"],
    ):
        a.imshow(img, interpolation="nearest", aspect="equal")
        a.set_title(ttl, pad=4)
        a.axis("off")

    if save_path:
        savefig(fig, save_path)
    else:
        plt.show()


def save_qualitative_examples(
    results:     Dict[str, dict],
    figure_dir:  str,
    cfg:         CFG,
    models:      Dict,
    input_stats: Dict,
    resolution:  str = "2mm",
    n_examples:  int = 4,
):
    if resolution not in results:
        return
    ds   = results[resolution]["dataset"]
    idxs = list(range(len(ds)))
    random.Random(0).shuffle(idxs)
    idxs = idxs[: min(n_examples, len(ds))]

    print(f"\n========== QUALITATIVE FIGURES ({resolution}) ==========")
    for k, i in enumerate(idxs, 1):
        vol = ds[i]
        safe_cuda_sync(cfg.device)
        t0 = time.time()
        if cfg.use_tta_in_testing:
            pred = predict_death_volume_tta(
                models[resolution], vol, cfg, input_stats[resolution],
                t_final=cfg.t_final, chunk=120000,
                flip_axes=cfg.tta_flip_axes,
            )
        else:
            pred = predict_death_volume(
                models[resolution], vol, cfg, input_stats[resolution],
                t_final=cfg.t_final, chunk=120000,
            )
        safe_cuda_sync(cfg.device)
        print(f"  Case {k}: {time.time() - t0:.3f}s")
        show_target_vs_output(
            vol, pred, cfg.channel_tags,
            title_suffix=f"  ({resolution})",
            save_path=os.path.join(
                figure_dir, f"qualitative_{resolution}_case{k:02d}"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Test Bioheat-PINN")
    p.add_argument("--test_dir",       type=str,  default="data/testing")
    p.add_argument("--checkpoint_dir", type=str,  default="checkpoints")
    p.add_argument("--figure_dir",     type=str,  default="figures")
    p.add_argument("--device",         type=str,  default=None)
    p.add_argument("--save_figures",   action="store_true")
    p.add_argument("--n_qualitative",  type=int,  default=4)
    p.add_argument("--resolutions",    nargs="+", default=["2mm", "3mm", "4mm"])
    p.add_argument("--no_tta",         action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    cfg = CFG(resolutions=tuple(args.resolutions))
    if args.device:
        cfg.device = args.device
    if args.no_tta:
        cfg.use_tta_in_testing = False

    if not os.path.isdir(args.test_dir):
        print(f"ERROR: test_dir not found: {args.test_dir}")
        sys.exit(1)

    # Load models from checkpoints
    models      = {}
    input_stats = {}
    for res in cfg.resolutions:
        print(f"Loading checkpoint for {res} ...")
        models[res], input_stats[res] = load_checkpoint_model(
            args.checkpoint_dir, res, cfg
        )

    # Run evaluation
    results = {}
    for res in cfg.resolutions:
        print(f"\n=== Evaluating {res} ===")
        results[res] = eval_resolution(
            args.test_dir, res, models[res], input_stats[res], cfg
        )
        r = results[res]
        print(
            f"  n={r['n']}  RMSE%={r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}  "
            f"time={r['time_mean']:.4f}s ± {r['time_std']:.4f}s  "
            f"fps≈{r['fps_mean']:.1f}"
        )

    # Print summary table
    print("\n==================== TEST SUMMARY ====================")
    print(
        f"{'Resolution':>10} | {'Avg RMSE%':>10} | "
        f"{'STD RMSE%':>10} | {'Avg time(s)':>12} | FPS"
    )
    print("-" * 62)
    for res in cfg.resolutions:
        r = results[res]
        print(
            f"{res:>10} | {r['rmse_mean']:>10.3f} | "
            f"{r['rmse_std']:>10.3f} | "
            f"{r['time_mean']:>12.4f} | {r['fps_mean']:.1f}"
        )

    # Save figures
    if args.save_figures:
        os.makedirs(args.figure_dir, exist_ok=True)
        print(f"\nSaving figures to: {args.figure_dir}")
        save_summary_figures(results, args.figure_dir, cfg.resolutions)
        save_qualitative_examples(
            results, args.figure_dir, cfg, models, input_stats,
            resolution=cfg.resolutions[0],
            n_examples=args.n_qualitative,
        )
        print("✅ Figures saved.")
