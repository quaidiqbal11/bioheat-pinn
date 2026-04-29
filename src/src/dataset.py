"""
dataset.py — Dataset classes for Bioheat-PINN.
Handles both training/validation and testing splits.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tifffile as tiff
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tifffile"])
    import tifffile as tiff

try:
    import imageio.v3 as iio
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio"])
    import imageio.v3 as iio


# ── Low-level volume loader ───────────────────────────────────────────────────
def load_vol(path: str) -> np.ndarray:
    """Load a 2-D or 3-D volume from .tif/.tiff or .png and return (D,H,W) float32."""
    p = path.lower()
    if p.endswith((".tif", ".tiff")):
        arr = tiff.imread(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        return arr
    if p.endswith(".png"):
        arr = iio.imread(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.max() > 1.5:
            arr /= 255.0
        return arr[None, :, :]
    raise ValueError(f"Unsupported file format: {path}")


# ── Training / Validation dataset ─────────────────────────────────────────────
class AblationVoxelDataset(Dataset):
    """
    Loads multi-channel volume tensors from the synthetic_* folder structure.
    Returns a tensor of shape (C, D, H, W) where channels follow `tags` order.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str],
        resolution: str,
        tags: Tuple[str, ...],
    ):
        self.split_path = (
            os.path.normpath(root)
            if split is None
            else os.path.join(root, split)
        )
        self.res  = resolution
        self.tags = tags

        if not os.path.isdir(self.split_path):
            raise FileNotFoundError(f"Split path not found: {self.split_path}")

        syn_folders = sorted(
            [d for d in os.listdir(self.split_path) if d.startswith("synthetic_")],
            key=lambda x: int(x.split("_")[1]),
        )
        self.samples: List[Dict[str, str]] = []
        for syn in syn_folders:
            res_dir = os.path.join(self.split_path, syn, resolution)
            if not os.path.isdir(res_dir):
                continue
            groups: Dict[Tuple[int, int], Dict[str, str]] = {}
            for fn in os.listdir(res_dir):
                if not fn.lower().endswith((".tif", ".tiff", ".png")):
                    continue
                base  = os.path.splitext(fn)[0]
                parts = base.split("_")
                if len(parts) < 4 or parts[0] != "synthetic":
                    continue
                try:
                    i, j = int(parts[1]), int(parts[2])
                except Exception:
                    continue
                tag = "_".join(parts[3:])
                if tag not in tags:
                    continue
                groups.setdefault((i, j), {})[tag] = os.path.join(res_dir, fn)
            for _, mp in groups.items():
                if all(t in mp for t in tags):
                    self.samples.append(mp)

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.split_path}/{resolution}.\n"
                f"Expected channel tags: {tags}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mp   = self.samples[idx]
        vols = {t: load_vol(mp[t]) for t in self.tags}
        dref = vols["applicator"].shape[0]
        if vols["slice"].shape[0] != dref:
            vols["slice"] = (
                np.repeat(vols["slice"], dref, axis=0)
                if vols["slice"].shape[0] == 1
                else vols["slice"][:dref]
            )
        x = np.stack([vols[t] for t in self.tags], axis=0).astype(np.float32)
        return torch.from_numpy(x)


# ── Testing dataset ───────────────────────────────────────────────────────────
def _parse_name(filename: str) -> Optional[Tuple[int, int, str]]:
    base  = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 4 or parts[0] != "synthetic":
        return None
    try:
        return int(parts[1]), int(parts[2]), "_".join(parts[3:])
    except Exception:
        return None


class TestVoxelDataset(Dataset):
    """Same structure as AblationVoxelDataset but for test folders (no labels required)."""

    def __init__(
        self,
        testing_dir: str,
        resolution: str,
        tags: Tuple[str, ...],
    ):
        self.tags = tags
        syn_folders = sorted(
            [d for d in os.listdir(testing_dir) if d.startswith("synthetic_")],
            key=lambda x: int(x.split("_")[1]),
        )
        self.samples: List[Dict[str, str]] = []
        for syn in syn_folders:
            res_dir = os.path.join(testing_dir, syn, resolution)
            if not os.path.isdir(res_dir):
                continue
            groups: Dict[Tuple[int, int], Dict[str, str]] = {}
            for fn in os.listdir(res_dir):
                if not fn.lower().endswith((".tif", ".tiff", ".png")):
                    continue
                parsed = _parse_name(fn)
                if not parsed:
                    continue
                i, j, tag = parsed
                if tag not in tags:
                    continue
                groups.setdefault((i, j), {})[tag] = os.path.join(res_dir, fn)
            for _, mp in groups.items():
                if all(t in mp for t in tags):
                    self.samples.append(mp)

        if not self.samples:
            raise RuntimeError(
                f"No test samples for {resolution} in {testing_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mp   = self.samples[idx]
        vols = {t: load_vol(mp[t]) for t in self.tags}
        dref = vols["applicator"].shape[0]
        if vols["slice"].shape[0] != dref:
            vols["slice"] = (
                np.repeat(vols["slice"], dref, axis=0)
                if vols["slice"].shape[0] == 1
                else vols["slice"][:dref]
            )
        return torch.from_numpy(
            np.stack([vols[t] for t in self.tags], axis=0).astype(np.float32)
        )
