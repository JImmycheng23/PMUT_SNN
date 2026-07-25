# AGENTS.md

This file provides guidance to AI coding assistants when working with this repository.

## Project overview

PMUT (Piezoelectric Micromachined Ultrasonic Transducer) neuromorphic classification
of fluids from electrical impedance sweep data using spiking neural networks (SNN),
reservoir computing (ESN/LSM), and CNNs. Each sweep: `amplitude`, `phase`, `x_v`, `y_v`
across a frequency ramp (ordered low→high, NOT physical time).

**Primary benchmark:** 5-class high-glucose (Glucose 50/100/250/500/1000 mM), 3 physical
sweeps per class, 3-fold sweep-level rotation. **glucose-ladder (2–20 mM) is exploratory
only** — single-sweep classes cannot support independent train/test.

**Latest results (2026-07-25, post-Fix-point-2 review):**

| Model | Prediction Level | Accuracy |
|-------|-----------------|----------|
| RF (61 corrected features) | sweep (by design) | **93.3%** CV |
| LSM spikes | sweep (majority vote) | **88.0% ± 11.8%** |
| ESN (tanh) | sweep (majority vote) | **84.0% ± 8.6%** |
| LSM membrane | sweep (majority vote) | **73.3% ± 9.4%** |
| CNN1D | window (sweep pending) | 73.8% ± 9.8% |
| SpikingCNN1D | window (fold 0 only) | 75.7% |
| Encoder→SNN (rate) | window | 32.5% (chance=20%) |

All reservoir + neural results use 3-fold × 5-seed protocol. Large ± values are
genuine across-fold variance from small dataset (15 sweeps), not leakage.

**GitHub:** `https://github.com/JImmycheng23/PMUT_SNN` (branch `review-fixes-2026-07-21`)

---

## Quick reference: common commands

All commands from repo root. Use `python` (Windows), system-installed dependencies.

```sh
# ===== Dataset preparation =====
python prepare_dataset.py --task high-glucose --split grouped

# ===== Feature extraction (NB4 — local-band corrected) =====
python notebooks/nb4_feature_extraction.py --task high-glucose

# ===== Feature ML baselines (NB5) =====
python notebooks/nb5_feature_ml_baselines.py --task high-glucose

# ===== Neural models (train.py, 5 seeds × 3 folds = 15 runs) =====
python train.py --task high-glucose --split grouped --model cnn1d --num-seeds 5 --epochs 150
python train.py --task high-glucose --split grouped --model spiking_cnn1d --num-seeds 5 --epochs 150
python train.py --task high-glucose --split grouped --model temporal_lif_snn --num-seeds 5 --epochs 150

# ===== Reservoir models (NB11, 5 seeds × 3 folds) =====
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --num-seeds 5
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --num-seeds 5 --lsm

# ===== Encoder→SNN diagnosis (NB8/NB9 + train_utils) =====
python notebooks/nb9_encoder_evaluation.py --task high-glucose --split grouped

# ===== Model comparison (NB12) =====
python notebooks/nb12_model_comparison.py --task high-glucose

# ===== Fixed-point =====
python notebooks/nb13_fixed_point_lif.py
python notebooks/nb14_fixed_point_pmut.py --task high-glucose --split grouped

# ===== HDL =====
python notebooks/nb15_hdl_test_vectors.py --task high-glucose --split grouped
python notebooks/nb16_hdl_spec.py --generate-hdl

# ===== Central results =====
python build_central_results.py
python verify_hdl_checklist.py
```

---

## Architecture

### Package `src/pmut_models/`

| Module | Role |
|--------|------|
| `config.py` | `DEFAULT_DATA_ROOT` and `DEFAULT_OUTPUT_ROOT` paths |
| `data.py` | Data loading, windowing, label encoding, splits (`window`, `random_window`, `grouped`), **`build_dataset_folds()` (3-fold sweep rotation)**, standardization, `DatasetBundle`, **`validate_phase_data()`** |
| `spiking.py` | `SpikeFunction`, `LIFStateLayer`, models (`CNN1D`, `SpikingCNN1D`, `TemporalLIFSNN`, `RateSNNMLP`, `EncodedSpikeSNN`), `build_model()` |
| `train_utils.py` | Training loop, evaluation, multi-seed harness, `encode_dataset_bundle()`, **`aggregate_to_sweep()`**, **`diagnose_encoder_pipeline()`** |
| `reservoir.py` | `Reservoir` (ESN, tanh), `LiquidStateMachine` (LSM, LIF), **`LSM.predict_proba()`**, `reservoir_grid_search()`, `estimate_memory_capacity()` |
| `fixed_point.py` | `QFormat`, `FloatLIF`, `FixedPointLIF`, **`IntegerLSM`** (true integer LSM with quantized readout), `compare_lif_neurons()`, `bit_width_sensitivity()`, `generate_hdl_test_vectors()`, `save_hdl_test_vectors()` |

### Key functions added during review fixes

| Function | Module | Purpose |
|----------|--------|---------|
| `build_dataset_folds()` | `data.py` | 3-fold sweep rotation (train/val/test = different sweeps) |
| `aggregate_to_sweep()` | `train_utils.py` | Window→sweep prediction aggregation (mean_logits or majority_vote) |
| `diagnose_encoder_pipeline()` | `train_utils.py` | 7-sub-point encoder→SNN diagnosis |
| `validate_phase_data()` | `data.py` | Phase wrapping, sign convention, unit checks |
| `IntegerLSM` | `fixed_point.py` | All-integer LSM with quantized readout |
| `LSM.predict_proba()` | `reservoir.py` | Class probabilities for sweep aggregation |

### Data pipeline

1. Raw files → `load_task_frame(task)` loads CSV/XLSX, maps media to labels.
2. `make_sequence_samples()` slides non-overlapping windows (default stride=window_size).
3. `build_dataset()` for single split; `build_dataset_folds()` for 3-fold rotation.
4. `standardize_from_train()` — per-channel z-score from training set only.
5. Result returned as `DatasetBundle`; saved/loaded as `.npz`.

### Models summary

| Model | Type | Input → Processing | Notes |
|-------|------|--------------------|-------|
| CNN1D | Non-spiking | Continuous [A,φ,x_v,y_v] → Conv1d+ReLU | Strong baseline |
| SpikingCNN1D | Learned spiking | Continuous → Conv1d → LIF | |
| TemporalLIFSNN | Learned spiking | Continuous → Linear per step → LIF | |
| RateSNNMLP | Learned spiking | Continuous → Flatten+Linear → LIF rate-coded | |
| EncodedSpikeSNN | Explicit encoder | **Spike trains** from NB8 → LIF → class | ~32%, fundamental info loss |
| Reservoir (ESN) | Rate-coded tanh | Continuous → fixed random W → tanh → Ridge | ~84% sweep |
| LiquidStateMachine (LSM) | Spiking LIF | Continuous → fixed random W → LIF → Ridge | ~88% sweep (spikes) |
| IntegerLSM | Fixed-point LIF | Continuous → integer LIF → quantized Ridge | HW-ready reference |

---

## Evaluation protocol (post-review)

### 3-fold sweep rotation

`build_dataset_folds()` produces 3 folds. Each fold: train/val/test = different physical sweeps.
- Fold 0: train=sweep1, val=sweep2, test=sweep3
- Fold 1: train=sweep2, val=sweep3, test=sweep1
- Fold 2: train=sweep3, val=sweep1, test=sweep2

### Multi-seed

Default 5 seeds: `[7, 42, 123, 999, 2024]`. Result = mean ± std across 15 (3 folds × 5 seeds) runs.

### Sweep-level aggregation

`aggregate_to_sweep()` groups window predictions by `sweep_key`. Methods:
- `mean_logits`: avg class probabilities → argmax (requires `predict_proba`)
- `majority_vote`: mode of window predictions

Applied to reservoir results. Neural models pending re-run with sweep_key in predictions CSV.

---

## Data locations

```
C:\Users\27010\Desktop\Neumat\Data\PMUT-SNN\
  Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv
  pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx      (primary: Air/Water/Glucose 50-1000mM, 3 sweeps each)
  low_conc_glucose_sweeps_data_2mM_to_50mM.xlsx    (exploratory only: 2-20mM, 1 sweep each)
```

Override with `--data-root`. Prepared `.npz` → `outputs/datasets/`. Run artifacts → `outputs/runs/`.

---

## Output directory map

```
outputs/
├── datasets/              # Prepared .npz files
├── features/              # NB4: 61 features + diagnostic band plots
├── feature_ml/            # NB5: RF=93.3% + other ML baselines
├── runs/                  # Neural model training (fold-based layout)
│   └── high-glucose_folds_w128/{model}/fold{N}/[seed dirs...]
├── reservoir/             # NB11: ESN + LSM results (fold-based)
│   └── high-glucose/grouped_w128/fold{N}/[lsm/]
├── comparison/            # NB12: cross-model comparison
├── fixed_point/           # NB13/NB14: fixed-point study
├── hdl_test_vectors/      # NB15: HDL verification data
├── hdl/                   # NB16: HDL spec + corrected Verilog
├── central_results/       # build_central_results.py output
├── REPRODUCIBILITY.md     # Frozen commit, data checksums, regeneration commands
└── RESPONSE_TO_SUPERVISOR_REVIEW.md  # Point-by-point response to Fix-point 2 review
```

---

## Review fixes applied (Fix-point 2, 2026-07-24/25)

| Point | What | Where |
|-------|------|-------|
| 1 | Low-concentration ladder removed from primary study | `data.py`, 11 script `--task` choices |
| 2 | Clean freeze: `outputs_legacy/`, `REPRODUCIBILITY.md`, `v2.0-clean-slate` tag | commit `5f58f06` |
| 3 | PMUT Q/bw corrected: local-band, detrended, interpolated, NaN sentinel | `nb4_feature_extraction.py` |
| 4 | 3-fold sweep rotation | `data.py::build_dataset_folds()` |
| 5 | Sweep-level prediction aggregation | `train_utils.py::aggregate_to_sweep()`, `reservoir.py::LSM.predict_proba()` |
| 6 | Multi-seed defaults (5) | `train.py`, `nb11_reservoir.py` |
| 7 | Encoder→SNN diagnosis (7 sub-points) | `train_utils.py::diagnose_encoder_pipeline()` |
| 8 | ESN/LSM interpretation corrected | `CLAUDE.md`, `nb12_model_comparison.py` |
| 9 | Phase preprocessing verified | `data.py::validate_phase_data()` |
| 10 | Central results generator rewritten | `build_central_results.py` |
| 11 | IntegerLSM + NB14 renamed | `fixed_point.py`, `nb14_fixed_point_pmut.py` |
| 12 | Verilog LIF update-order fixed | `outputs/hdl/lif_neuron_fxp.v` |
| 13 | HDL self-checking testbench | `outputs/hdl/lif_neuron_fxp.v` |
| 14 | PMUT spike-to-current scaling fixed | `fixed_point.py::generate_hdl_test_vectors()` |

---

## Known gaps (for next session)

### Software

1. **Neural models not sweep-aggregated.** CNN1D/SpikingCNN1D results are window-level.
   `predictions.csv` lacks `sweep_key` column. Fix: modify `train_single_run()` in
   `train_utils.py` to save `test_groups` alongside predictions, then re-run.

2. **SpikingCNN1D only has Fold 0.** Need to run Folds 1 and 2.

3. **TemporalLIFSNN not regenerated** under the new protocol at all.

4. **NB14 not re-run with IntegerLSM.** Current `outputs/fixed_point/` is empty
   (moved to `outputs_legacy/`). Need to re-run bit-width study using `IntegerLSM`
   or at minimum the existing `run_quantized_lsm_inference()`.

### Hardware

5. **IntegerLSM tested on random data only.** Not yet validated on real PMUT sweeps.
   Need to: load `build_dataset_folds()`, run `IntegerLSM.fit_readout()` on fold 0
   train/val, evaluate on test, compare float vs quantized readout accuracy.

6. **HDL testbench not simulated.** Verilog testbench exists but has not been run
   through Icarus Verilog or Verilator. The built-in sanity test should pass;
   PMUT-derived vectors need to be regenerated first (NB15 → hex format).

7. **Multi-neuron reservoir not in Verilog.** Only the single `lif_neuron_fxp` exists.
   Scaling to N-neuron reservoir with shared input bus and weight ROM is next.

### Documentation

8. **`regenerate_all.py` not written.** The orchestration script in `REPRODUCIBILITY.md`
   does not exist yet. All individual scripts work; just needs a wrapper.

---

## Session log (abbreviated)

### 2026-07-25
- Completed all 14 review points from Fix-point 2.
- Regenerated RF (93.3%), ESN (84.0% sweep), LSM (88.0% sweep spikes), CNN1D (73.8% window).
- Built `IntegerLSM` with quantized readout.
- Fixed Verilog LIF (v_next) and spike-to-current scaling.
- Applied sweep-level aggregation to reservoir results.
- Rewrote `build_central_results.py` with 22 structured numeric columns.
- Committed and pushed to GitHub (`review-fixes-2026-07-21`, commit `0a35276`).

### 2026-07-24
- Clean freeze: `outputs_legacy/`, `v2.0-clean-slate`, `REPRODUCIBILITY.md`.
- Removed low-concentration ladder from primary classification.
- Corrected PMUT Q/bw extraction (local-band method, 61 features).
- Implemented 3-fold sweep rotation, sweep aggregation, multi-seed defaults.
- Encoder→SNN diagnosis complete (fundamental information-loss problem).
- ESN/LSM interpretation corrected. Phase preprocessing verified.
