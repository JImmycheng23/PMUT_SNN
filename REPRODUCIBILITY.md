# Reproducibility Reference

**Frozen commit:** `d64c887` (branch: `review-fixes-2026-07-21`)
**Tag:** `v2.0-final`
**Date:** 2026-07-28
**Python:** 3.13.7

## Raw data files

| File | MD5 |
|------|-----|
| `pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx` | `397DC1F0904D471B844DEFABDC9021A7` |
| `low_conc_glucose_sweeps_data_2mM_to_50mM.xlsx` | `A4EDC1E3B1BF8EF71BA4D3A4C09A21AD` |
| `Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv` | `F284E4A0F7B1DAB583C7B40CD22F8F98` |

**Data root:** `C:\Users\27010\Desktop\Neumat\Data\PMUT-SNN\`

## Primary benchmark

**Task:** `high-glucose` — 5-class (Glucose 50/100/250/500/1000 mM), 3 sweeps/class, grouped split

The `glucose-ladder` task is **exploratory only** (see Point 1 of Fix-point 2 review).
Low-concentration data (2–20 mM) have only 1 sweep each — no sweep-level train/test possible.

## Quick regeneration (single command)

```sh
python regenerate_all.py --task high-glucose --window-size 128 --num-seeds 5 --epochs 150
```

Use `--skip-neural`, `--skip-reservoir`, or `--skip-features` to skip compute-heavy stages.

## Step-by-step regeneration commands

All commands run from repo root. Outputs go to `outputs/`.

```sh
# Phase A: Prepare dataset
python prepare_dataset.py --task high-glucose --split grouped

# Phase B: Feature extraction (NB4) — all 15 sweeps
python notebooks/nb4_feature_extraction.py --task high-glucose

# Phase C: Feature ML baselines (NB5)
python notebooks/nb5_feature_ml_baselines.py --task high-glucose

# Phase D: Neural models (5 seeds × 3 folds = 15 runs each)
python train.py --task high-glucose --folds --model cnn1d --num-seeds 5 --epochs 150
python train.py --task high-glucose --folds --model spiking_cnn1d --num-seeds 5 --epochs 150
python train.py --task high-glucose --folds --model temporal_lif_snn --num-seeds 5 --epochs 150
python train.py --task high-glucose --folds --model rate_snn_mlp --num-seeds 5 --epochs 150

# Phase E: Reservoir models (5 seeds × 3 folds)
python notebooks/nb11_reservoir.py --task high-glucose --folds --num-seeds 5
python notebooks/nb11_reservoir.py --task high-glucose --folds --num-seeds 5 --lsm

# Phase F: Encoder evaluation (NB9)
python notebooks/nb9_encoder_evaluation.py --task high-glucose --split grouped

# Phase G: Model comparison (NB12)
python notebooks/nb12_model_comparison.py --task high-glucose

# Verification
python build_central_results.py
python verify_hdl_checklist.py
```

## Key dependencies

```
numpy, scipy, pandas, scikit-learn, matplotlib, openpyxl
torch (for neural models)
```

## Output directory map

```
outputs/
├── datasets/              # Prepared .npz files
├── features/              # NB4: 61 features + 15 diagnostic band plots
├── feature_ml/            # NB5: RF + other ML baselines
├── runs/                  # Neural model training (fold-based layout)
│   └── high-glucose_folds_w128/{model}/fold{N}/[seed dirs...]
├── reservoir/             # NB11: ESN + LSM results (fold-based)
│   └── high-glucose/grouped_w128/fold{N}/[lsm/]
├── comparison/            # NB12: cross-model comparison
├── central_results/       # build_central_results.py output
└── REPRODUCIBILITY.md     # This file
```

## Legacy outputs

Pre-freeze outputs are preserved in `outputs_legacy/`. They contain results from
mixed code versions and should NOT be used for the final manuscript.
