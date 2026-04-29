# Bioheat-PINN

**Physics-Informed Neural Network for Thermal Ablation Field Prediction**

Official implementation of the paper:

> **"Bioheat-PINN: A Physics-Informed Neural Network Surrogate for Predicting Thermal Ablation Fields"**
> *[Authors], [Venue], [Year]*

---

## Overview

Bioheat-PINN is a hybrid surrogate model that combines a multi-layer perceptron (MLP) with physics-based regularization through the Pennes bioheat equation. It predicts volumetric thermal death fields at multiple voxel resolutions (2 mm, 3 mm, 4 mm) from tissue property maps, enabling fast and accurate ablation planning.

```
L = L_data + L_pde + L_ic + L_bc
```

The model is trained at three voxel resolutions independently, with a composite early-stopping criterion that penalizes both mean RMSE% and per-case standard deviation.

---

## Results

| Method            | Resolution (mm) | Avg. RMSE (%) | STD (%) |
|-------------------|:--------------:|:-------------:|:-------:|
| FDM               | 3              | 1.63          | 0.38    |
|                   | 4              | 1.88          | 0.43    |
| C-NCA             | 2              | 1.43          | 0.76    |
|                   | 3              | 1.79          | 0.97    |
|                   | 4              | 1.74          | 0.77    |
| **Bioheat-PINN**  | **2**          | **1.26**      | **0.43**|
| **(Ours)**        | **3**          | **1.43**      | **0.34**|
|                   | **4**          | **1.57**      | **0.32**|

> FDM at 2 mm did not converge to a stable solution and is excluded.

---

## Repository Structure

```
bioheat_pinn/
├── src/
│   ├── config.py          # CFG dataclass and hyperparameters
│   ├── dataset.py         # Dataset loaders (train / val / test)
│   ├── model.py           # HybridContextPINN architecture
│   ├── losses.py          # Physics residual, data, IC/BC losses
│   ├── utils.py           # Normalisation, feature pyramid, sampling
│   ├── train.py           # Training entry point
│   └── test.py            # Testing / evaluation entry point
├── checkpoints/           # Saved model checkpoints (auto-created)
├── figures/               # Output figures (auto-created)
├── data/                  # Place dataset folders here (see below)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/bioheat_pinn.git
cd bioheat_pinn
pip install -r requirements.txt
```

---

## Dataset

The dataset follows this folder structure:

```
data/
├── training/
│   ├── synthetic_1/
│   │   ├── 2mm/
│   │   │   ├── synthetic_1_0_applicator.tif
│   │   │   ├── synthetic_1_0_perfusion.tif
│   │   │   ├── synthetic_1_0_density.tif
│   │   │   ├── synthetic_1_0_specific_heat.tif
│   │   │   ├── synthetic_1_0_slice.tif
│   │   │   └── synthetic_1_0_death.tif
│   │   ├── 3mm/
│   │   └── 4mm/
│   └── synthetic_2/ ...
├── validation/
│   └── synthetic_*/ ...
└── testing/
    └── synthetic_*/ ...
```

Update the paths at the top of `src/train.py` and `src/test.py` to point to your data directories, or pass them via command-line arguments.

---

## Training

```bash
python src/train.py \
    --train_dir data/training \
    --val_dir   data/validation \
    --epochs    140 \
    --device    cuda
```

Checkpoints are saved automatically to `checkpoints/` for the best model at each resolution.

---

## Testing

```bash
python src/test.py \
    --test_dir    data/testing \
    --checkpoint_dir checkpoints \
    --device      cuda \
    --save_figures
```

Figures are saved to `figures/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{bioheatpinn2025,
  title   = {Bioheat-PINN: A Physics-Informed Neural Network Surrogate
             for Predicting Thermal Ablation Fields},
  author  = {[Authors]},
  journal = {[Journal/Conference]},
  year    = {2025},
}
```

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.
