"""
config.py — CFG dataclass and global hyperparameters for Bioheat-PINN.
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ── Device auto-detection ─────────────────────────────────────────────────────
def probe_best_device() -> str:
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU.")
        return "cpu"
    try:
        idx          = torch.cuda.current_device()
        name         = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
    except Exception as e:
        print(f"CUDA probe error ({e}) — using CPU.")
        return "cpu"
    if major < 7:
        print(f"GPU {name} (sm_{major}{minor}) unsupported — using CPU.")
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda"); _ = x + 1
        del x; torch.cuda.synchronize()
        print(f"Using CUDA: {name} (sm_{major}{minor})")
        return "cuda"
    except Exception as e:
        print(f"CUDA probe failed ({e}) — using CPU.")
        return "cpu"


ACTIVE_DEVICE = probe_best_device()


# ── Main config ───────────────────────────────────────────────────────────────
@dataclass
class CFG:
    # Paths
    root:        str = "data"
    split_train: str = "training"
    split_val:   str = "validation"
    train_dir:   Optional[str] = None
    val_dir:     Optional[str] = None
    testing_dir: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    figure_dir:     str = "figures"

    # Data
    resolutions:  Tuple[str, ...] = ("2mm", "3mm", "4mm")
    channel_tags: Tuple[str, ...] = (
        "applicator", "perfusion", "density",
        "specific_heat", "slice", "death",
    )

    # Runtime
    device: str = ACTIVE_DEVICE
    seed:   int = 42

    # Training schedule
    batch_volumes:  int   = 1
    epochs:         int   = 140
    warmup_epochs:  int   = 8
    lr:             float = 3e-4
    min_lr:         float = 3e-5
    weight_decay:   float = 1e-5
    beta1:          float = 0.9
    beta2:          float = 0.999
    eps:            float = 1e-8
    grad_clip:      float = 1.0
    ema_decay:      float = 0.997

    # Physics
    t_final: float = 1.0
    Tb:      float = 0.0

    # Sampling
    n_data_pts:    int = 16384
    n_collocation: int = 2048
    n_ic_pts:      int = 2048
    n_bc_pts:      int = 2048

    # Tissue conductivity model
    k0: float = 0.02
    k1: float = 0.10
    q0: float = 1.0

    # Loss weights
    w_data:            float = 1.0
    w_pde:             float = 0.08
    w_ic:              float = 0.03
    w_bc:              float = 0.01
    pde_warmup_epochs: int   = 50
    bc_warmup_epochs:  int   = 25

    # Importance sampling
    focus_frac:          float = 0.80
    sample_w_death:      float = 2.50
    sample_w_boundary:   float = 2.25
    sample_w_applicator: float = 1.25

    # Death-field loss terms
    pos_weight_strength:      float = 4.0
    boundary_weight_strength: float = 3.0
    bce_weight:   float = 0.20
    dice_weight:  float = 0.20
    huber_weight: float = 0.15
    mse_weight:   float = 0.25
    topk_weight:  float = 0.20
    focal_gamma:  float = 1.5

    # Validation
    stats_max_volumes:    int   = 250
    val_subset:           int   = 60
    val_score_std_weight: float = 0.25

    # Architecture
    model_width:     int = 320
    model_depth:     int = 10
    n_fourier_freqs: int = 4
    context_pool_ks: Tuple[int, ...] = (1, 3, 7)

    # TTA
    use_tta_in_validation: bool = True
    use_tta_in_testing:    bool = True
    tta_flip_axes: Tuple[Tuple[int, ...], ...] = ((), (1,), (2,), (3,))
