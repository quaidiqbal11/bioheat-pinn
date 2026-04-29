"""
model.py — HybridContextPINN architecture for Bioheat-PINN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Gated residual block with LayerNorm and GLU-style activation."""

    def __init__(self, width: int):
        super().__init__()
        self.ln  = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.ln(x)
        a, b = self.fc1(h).chunk(2, dim=1)
        h    = F.gelu(a) * torch.sigmoid(b)
        return x + 0.5 * self.fc2(h)


class HybridContextPINN(nn.Module):
    """
    Physics-Informed MLP surrogate with:
      - Fourier feature encoding of (x, y, z, t)
      - Multi-scale context features from tissue property volumes
      - Dual output heads: temperature field T and death probability D
      - Learnable conductivity and perfusion scaling parameters
    """

    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, width)
        self.blocks   = nn.ModuleList(
            [ResidualMLPBlock(width) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(width)

        # Temperature head
        self.head_T = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        # Death probability head (conditioned on T)
        self.head_D = nn.Sequential(
            nn.Linear(width + 1, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

        # Learnable physics scaling parameters
        self.log_k_scale    = nn.Parameter(torch.tensor(0.0))
        self.log_perf_scale = nn.Parameter(torch.tensor(0.0))

    def k_multiplier(self) -> torch.Tensor:
        return torch.exp(self.log_k_scale)

    def perf_multiplier(self) -> torch.Tensor:
        return torch.exp(self.log_perf_scale)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)

    def forward_with_logits(
        self, x: torch.Tensor
    ):
        """Returns (t_pred, d_prob, d_logits) — used during training for loss computation."""
        h        = self.trunk(x)
        t_pred   = self.head_T(h)
        d_logits = self.head_D(torch.cat([h, t_pred], dim=1))
        return t_pred, torch.sigmoid(d_logits), d_logits

    def forward(self, x: torch.Tensor):
        """Returns (t_pred, d_prob) — used during inference."""
        t, d, _ = self.forward_with_logits(x)
        return t, d
