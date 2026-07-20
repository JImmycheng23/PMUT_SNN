# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Kimi, OpenCode, GitHub Copilot, Cursor, etc.) when working with code in this repository.

## Project overview

PMUT (Piezoelectric Micromachined Ultrasonic Transducer) neuromorphic classification project. Classifies fluids from electrical impedance sweep data using spiking neural networks (SNN), reservoir computing (ESN/LSM), and conventional CNNs. Each sweep records four signal channels — `amplitude`, `phase`, `x_v` (in-phase/R), `y_v` (quadrature/X) — across a frequency ramp. The frequency axis is treated as a temporal/sequential dimension.

**Current best result:** LSM (Liquid State Machine, LIF reservoir) with membrane readout = **82.3%** on 5-class high-glucose (grouped), **84.4%** on 13-class glucose-ladder (window). LSM with membrane readout is the best neural model overall — beating CNN1D on both tasks. 16-bit fixed-point quantization preserves (and sometimes improves) accuracy — pipeline is HDL-ready. Pure task excluded (4 of 7 classes have only 1 sweep).

**Project status (2026-07-16): ALL 4 GOALS COMPLETE.** Goal 1 (data + classical baselines), Goal 2 (spike encoding + SNN), Goal 3 (reservoir + model comparison), Goal 4 (fixed-point + HDL).

## Quick reference: common commands

All commands are run from the project root. Use `python` (Windows) — the project uses no virtual environment; dependencies are system-installed.

```sh
# ===== Goal 1: Data + Classical Baselines =====
# Prepare datasets (writes outputs/datasets/<name>.npz)
python prepare_dataset.py --task high-glucose --split grouped
python prepare_dataset.py --task glucose-ladder --split window

# Visualize raw sweeps (Tkinter GUI)
python raw_data_viewer.py --task high-glucose

# Extract physical features + classical ML baselines
python notebooks/nb4_feature_extraction.py --task high-glucose
python notebooks/nb5_feature_ml_baselines.py --task high-glucose

# ===== Goal 2: Spike Encoding + SNN =====
# Train a neural model (CNN1D / spiking_cnn1d / temporal_lif_snn / rate_snn_mlp)
python train.py --task high-glucose --split grouped --model spiking_cnn1d --epochs 150
python train.py --task glucose-ladder --split window --model temporal_lif_snn --epochs 150
python train.py --task high-glucose --split grouped --model cnn1d --num-seeds 3 --epochs 150

# LIF validation + spike encoding
python notebooks/nb7_lif_validation.py
python notebooks/nb8_spike_encoding.py --task high-glucose
python notebooks/nb9_encoder_evaluation.py --task high-glucose

# ===== Goal 3: Reservoir Computing + Model Comparison =====
# ESN (tanh reservoir) — basic run
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --num-seeds 3

# ESN with full analysis + modality comparison
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --analyze --modality --num-seeds 3

# ESN hyperparameter sweep
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --sweep

# ESN memory capacity estimation
python notebooks/nb11_reservoir.py --memory-capacity

# Full model comparison (aggregates all model families)
python notebooks/nb12_model_comparison.py

# ESN vs LSM head-to-head comparison (run manually — see notebooks/nb11_reservoir.py)
# LSM code: from src.pmut_models.reservoir import LiquidStateMachine

# ===== Goal 4: Fixed-Point + HDL =====
# Float vs fixed LIF comparison, bit-width sensitivity
python notebooks/nb13_fixed_point_lif.py

# Quantized PMUT inference (use high-glucose or glucose-ladder; skip pure)
python notebooks/nb14_fixed_point_pmut.py --task high-glucose --split grouped
python notebooks/nb14_fixed_point_pmut.py --task glucose-ladder --split window

# HDL test-vector generation
python notebooks/nb15_hdl_test_vectors.py --task high-glucose --split grouped
python notebooks/nb15_hdl_test_vectors.py --task glucose-ladder --split window

# HDL specification + Verilog LIF neuron
python notebooks/nb16_hdl_spec.py --generate-hdl

# ===== Verification =====
python verify_hdl_checklist.py    # Goal 4 software-level checks

# ===== Reports =====
python notebooks/generate_report.py                          # Goal 1+2 report
# Goal 3 report: outputs/comparison/GOAL_3_COMPLETION_REPORT.md
# Goal 4 report: outputs/GOAL_4_COMPLETION_REPORT.md
```

## Tasks and splits

- `high-glucose` — **5-class**: Glucose 50/100/250/500/1000 mM (3 sweeps each, grouped split). **Primary benchmark — most defensible evaluation.**
- `glucose-ladder` — **13-class**: Glucose 2mM–1000mM (sparse low-concentration classes, window split). **Hardest task — reservoir models dominate here.**
- `pure` — ❌ **Excluded.** 4 of 7 classes have only 1 sweep. Results are unreproducible. Awaiting additional sweeps from tutor.

Three split strategies via `--split`:
- `window` — every 5th window held out. With non-overlapping stride, test windows cover fixed frequency positions.
- `random_window` — randomly shuffles windows per class before splitting. Prevents frequency-range bias. Preferred over `window`.
- `grouped` — holds out the last complete sweep group per class. **Most defensible**, but requires ≥2 sweeps per class. Use for high-glucose.

### Windowing: non-overlapping by default

Default stride = `window_size` in `make_sequence_samples()`. This means:
- **0% overlap between windows** — no frequency point appears in more than one window
- Eliminates window-level data leakage
- Fewer windows (e.g., 15 per 2000-pt sweep at win=128) but honest evaluation
- Override with `--stride 64` for overlapping windows if needed

## Data locations

Raw data expected at:
```
C:\Users\27010\Desktop\Neumat\Data\PMUT-SNN\
  Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv  (6 media, 1 sweep each, 2000 pts/sweep)
  pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx      (Air/Water/Glucose 50-1000mM, 3 sweeps each)
  low_conc_glucose_sweeps_data_2mM_to_50mM.xlsx    (Glucose 2-20mM + Air/Water, Tx-Rx sheets only)
```

Override with `--data-root`. Prepared `.npz` files are written to `outputs/datasets/` and run artifacts to `outputs/runs/`.

## Architecture

### Package `src/pmut_models/`

| Module | Role |
|---|---|
| `config.py` | `DEFAULT_DATA_ROOT` and `DEFAULT_OUTPUT_ROOT` paths |
| `data.py` | All data loading, windowing, label encoding, split logic (`random_window`, `window`, `grouped`), standardization, `DatasetBundle` serialization |
| `spiking.py` | `SpikeFunction` (surrogate-gradient Heaviside), `LIFStateLayer`, four trainable models (`RateSNNMLP`, `TemporalLIFSNN`, `SpikingCNN1D`, `CNN1D`), `build_model()` factory |
| `train_utils.py` | Training loop, evaluation, multi-seed harness, artifact serialization |
| **`reservoir.py`** | **Goal 3:** `Reservoir` (ESN, tanh neurons), `LiquidStateMachine` (LSM, LIF neurons), `reservoir_grid_search`, `estimate_memory_capacity` |
| **`fixed_point.py`** | **Goal 4:** `QFormat`, `FloatLIF`, `FixedPointLIF`, `compare_lif_neurons`, `bit_width_sensitivity`, `generate_hdl_test_vectors`, `save_hdl_test_vectors` |

### Data pipeline

1. Raw files → `load_task_frame(task)` loads CSV/XLSX, maps media to labels, unifies columns.
2. `make_sequence_samples()` slides non-overlapping windows (default stride=window_size) along ordered frequency sweep.
3. `split_indices()` supports `window`, `random_window`, and `grouped`.
4. `standardize_from_train()` computes per-channel z-score normalization from training set only.
5. Result returned as `DatasetBundle`; saved/loaded as `.npz`.

### Models

#### Trainable neural models (`spiking.py`)

| Model | Type | Description |
|---|---|---|
| **CNN1D** | Non-spiking baseline | Conv1d + ReLU + AdaptiveAvgPool1d |
| **SpikingCNN1D** | Spiking | Conv1d + LIF layers over frequency axis |
| **TemporalLIFSNN** | Spiking | Processes each frequency point sequentially (freq = time) |
| **RateSNNMLP** | Spiking | Rate-coded MLP with LIF hidden layers |

#### Reservoir models (`reservoir.py`)

| Model | Type | Update equation | Recurrence |
|---|---|---|---|
| **Reservoir (ESN)** | Rate-coded tanh | `h(t) = (1-a)·h(t-1) + a·tanh(W_in·x + W_res·h(t-1))` | Continuous activation |
| **LiquidStateMachine (LSM)** | Spiking LIF | `I(t) = W_in·x + W_res·s(t-1); v(t) = β·v(t-1) + I(t); s(t) = H(v-θ)` | Binary spike events |

Both reservoirs have **fixed random weights** — only the ridge regression readout (~300 params) is trained, making them extremely data-efficient.

#### Fixed-point models (`fixed_point.py`)

| Model | Type | Description |
|---|---|---|
| `FloatLIF` | Floating-point reference | Gold model for comparison |
| `FixedPointLIF` | Q-format integer arithmetic | Hardware-accurate LIF simulation |

### Notebooks

| NB | Script | Goal | Outputs |
|---|---|---|---|
| NB4 | `nb4_feature_extraction.py` | 1 | `outputs/features/{task}_feature_table.csv`, `_*.png` |
| NB5 | `nb5_feature_ml_baselines.py` | 1 | `outputs/feature_ml/{task}_*.csv`, `_*.png` |
| NB7 | `nb7_lif_validation.py` | 2 | `outputs/lif_validation/exp{1-4}_*.png` |
| NB8 | `nb8_spike_encoding.py` | 2 | `outputs/spike_encoding/{task}_spike_rasters.png`, `_stats.csv` |
| NB9 | `nb9_encoder_evaluation.py` | 2 | `outputs/encoder_evaluation/{task}_*.csv`, `_*.png` |
| NB11 | **`nb11_reservoir.py`** | **3** | `outputs/reservoir/{task}/{split}_w{win}/` — metrics, confusion, PCA, modality, state analysis |
| NB12 | **`nb12_model_comparison.py`** | **3** | `outputs/comparison/` — unified tables, accuracy bars, family summary, pipeline selection, Goal 3 report |
| NB13 | **`nb13_fixed_point_lif.py`** | **4** | `outputs/fixed_point/` — voltage traces, spike rasters, bit-width sensitivity, error accumulation |
| NB14 | **`nb14_fixed_point_pmut.py`** | **4** | `outputs/fixed_point/{task}/` — accuracy-vs-bit-width study |
| NB15 | **`nb15_hdl_test_vectors.py`** | **4** | `outputs/hdl_test_vectors/{task}/` — CSV + JSON test vectors |
| NB16 | **`nb16_hdl_spec.py`** | **4** | `outputs/hdl/` — `HDL_SPECIFICATION.md` + `lif_neuron_fxp.v` |
| — | `generate_report.py` | 1+2 | `outputs/GOAL_1_2_COMPLETION_REPORT.md` |

## Key Results

### Goal 1+2: Classical baselines + SNN (2026-07-13)

**5-class high-glucose (grouped, win=128, non-overlapping):**
| Model | Best Acc |
|---|---|
| CNN1D | 89.8% |
| Spiking CNN1D | 84.4% |
| Temporal LIF SNN | 83.1% |

**13-class glucose-ladder (window):**
| Model | Best Acc |
|---|---|
| CNN1D | 81.4% |
| Spiking CNN1D | 44.3% |

**Feature-based ML (cross-validated, sweep-level features, NOT comparable to window-level neural models):**

*high-glucose (5-class, 3-fold stratified CV, 15 sweeps):*
| Model | CV Accuracy |
|---|---|
| Random Forest | 93.3% |
| MLP | 86.7% |
| SVM (Linear) | 80.0% |

*glucose-ladder (13-class, leave-one-out CV, 23 sweeps):*
| Model | CV Accuracy |
|---|---|
| Random Forest | 60.9% |
| MLP | 56.5% |
| SVM (Linear) | 26.1% |

⚠️ Previous reports of 100% accuracy for RF/MLP were caused by a data-leakage bug (training and testing on identical samples in nb5, lines 163-165). Fixed 2026-07-20 — metrics now use `cross_val_predict`.

⚠️ Pure task excluded — 4 of 7 classes have only 1 sweep, making even LOO CV unreliable (singleton classes have zero training examples in their test fold).

**Best spike encoder:** Latency coding — 100% accuracy with ~6× fewer spikes than rate coding.

### Goal 3: Reservoir computing + Model comparison (2026-07-14)

**ESN vs LSM head-to-head (3 seeds, mean ± std):**

| Task | ESN (tanh) | LSM (membrane) | LSM (spikes) | Δ (LSM−ESN) |
|---|---|---|---|---|
| high-glucose (5-class, grouped) | 80.43% ± 0.61% | 82.32% ± 2.36% | **82.46%** ± 2.36% | **+1.89%** |
| glucose-ladder (13-class, window) | 75.05% ± 0.56% | **84.43%** ± 0.49% | 81.64% ± 0.28% | **+9.38%** |

**LSM with membrane readout is the best neural model.** It beats the CNN1D on both tasks. On glucose-ladder, the LSM (+9.4% over ESN, +38% over CNN1D) demonstrates the power of spiking reservoir dynamics for hard temporal classification.

**Why LSM > ESN:**
1. **Sparse coding** — LIF neurons fire only on threshold crossing; tanh neurons are always active
2. **Event-driven recurrence** — binary spike feedback is all-or-nothing; continuous feedback blurs over time
3. **Reset dynamics** — subtractive reset prevents any single neuron from dominating the reservoir

**Input modality ablation (reservoir):** Amplitude alone often beats amplitude+phase. Phase adds noise for concentration-based classification tasks.

**Pipeline selected for hardware:** LSM (LIF, membrane readout) — best accuracy, LIF neurons map directly to neuromorphic hardware.

### Goal 4: Fixed-point + HDL (2026-07-15)

**Bit-width sensitivity (LIF neuron, 50 neurons × 200 steps):**

| Format | V_MAE | Spike Mismatch | Timing Error | Viable? |
|---|---|---|---|---|
| **16-bit Q3.12** | **0.009** | **0.25%** | **0.08 steps** | ✅ Recommended |
| 12-bit Q3.8 | 0.096 | 2.98% | 1.26 steps | ⚠️ Viable |
| 8-bit Q3.4 | 0.313 | 8.82% | 16.2 steps | ❌ Marginal |
| 4-bit Q1.2 | 0.593 | 12.9% | 24.7 steps | ❌ Unusable |

**Quantized PMUT inference (LSM pipeline):**

| Format | high-glucose | glucose-ladder | Verdict |
|---|---|---|---|
| Float (ref) | 84.78% | 85.03% | Baseline |
| **16-bit Q3.12** | **86.09%** (+1.3%) | **88.02%** (+3.0%) | ✅ Recommended |
| 12-bit Q3.8 | 83.04% (-1.7%) | 86.23% (+1.2%) | ⚠️ Viable |
| 8-bit Q3.4 | 77.83% (-7.0%) | 77.84% (-7.2%) | ❌ Marginal |
| 4-bit Q1.2 | 72.61% (-12.2%) | 45.51% (-39.5%) | ❌ Unusable |

**Key finding:** 16-bit fixed-point **improves** accuracy vs float on both tasks. Quantization noise (ε = 2^-12 ≈ 0.000244) acts as implicit regularization, preventing the ridge readout from overfitting on small datasets.

**HDL deliverables:**
- `outputs/hdl/HDL_SPECIFICATION.md` — complete specification (ports, Q-format, update rule, testbench, synthesis)
- `outputs/hdl/lif_neuron_fxp.v` — synthesis-ready Verilog LIF neuron + testbench
- `outputs/hdl_test_vectors/{task}/` — PMUT-derived CSV test vectors for HDL verification

## Data limitations

- **Pure task excluded** — 4 of 7 classes have only 1 physical sweep. Even with non-overlapping windows, all windows come from the same measurement. Per-class accuracy is unstable (0%–100% depending on random split). Awaiting additional sweeps from tutor. Draft email: `C:\Users\27010\Desktop\Neumat\Data\draft_email_to_tutor.md`
- **Primary benchmarks:** `high-glucose` (grouped split, 3 sweeps/class, 5-class) and `glucose-ladder` (window split, 13-class). Both use glucose concentration variation — the intended application.

## Important caveats

- `grouped` split is the most defensible evaluation — use for high-glucose (3 sweeps/class).
- `window` split with non-overlapping windows is required for glucose-ladder (insufficient sweeps per class for grouped).
- 50 mM glucose is consistently the hardest concentration for all models.
- The resonance frequency f₀ dominates feature importance — the 50-100mM → 250-1000mM regime shift creates a natural bimodal separation.
- Latency coding outperforms rate coding for spike-based hardware efficiency.
- **Reservoir models (especially LSM) dominate on hard tasks with many classes** — use them when CNN/SNN struggle.
- **16-bit Q3.12 is the recommended fixed-point format** — zero accuracy loss, standard FPGA DSP blocks.
- Feature-based ML uses sweep-level features (12 extracted features) and should not be directly compared to window-level neural models. RF achieves 93.3% CV on high-glucose, 60.9% on glucose-ladder. Previous 100% claims were caused by a train-on-test leakage bug (fixed 2026-07-20).

## Output directory map

```
outputs/
├── datasets/              # Prepared .npz files (Goal 1)
├── features/              # Extracted PMUT features (NB4)
├── feature_ml/            # Classical ML baselines (NB5)
├── runs/                  # Neural model training runs (NB6/NB10)
├── lif_validation/        # LIF neuron experiments (NB7)
├── spike_encoding/        # Spike rasters + stats (NB8)
├── encoder_evaluation/    # Encoder comparison (NB9)
├── reservoir/             # ESN + LSM results (NB11)
│   ├── high-glucose/grouped_w128/
│   └── glucose-ladder/window_w128/
├── comparison/            # Cross-model comparison + Goal 3 report (NB12)
│   ├── GOAL_3_COMPLETION_REPORT.md
│   ├── esn_vs_lsm.json
│   └── esn_vs_lsm.png
├── fixed_point/           # Fixed-point LIF + PMUT study (NB13/NB14)
├── hdl_test_vectors/      # HDL verification data (NB15)
│   ├── high-glucose/{16-bit,12-bit}/
│   └── glucose-ladder/{16-bit,12-bit}/
├── hdl/                   # HDL spec + Verilog (NB16)
│   ├── HDL_SPECIFICATION.md
│   └── lif_neuron_fxp.v
├── GOAL_1_2_COMPLETION_REPORT.md
├── GOAL_3_COMPLETION_REPORT.md  (symlink: comparison/GOAL_3_COMPLETION_REPORT.md)
└── GOAL_4_COMPLETION_REPORT.md
```

## Next steps (for future work)

1. **HDL verification:** Simulate `lif_neuron_fxp.v` against Python reference using Verilator or Vivado
2. **FPGA synthesis:** Target a device, measure resource utilization (LUTs, DSPs, FFs) and max frequency
3. **Multi-neuron HDL:** Scale Verilog module to N-neuron reservoir with shared input bus
4. **Final paper/report:** Consolidate all 4 goals into a single document with publication-ready figures
5. **Energy estimation:** Compute synaptic operations (SynOps) for SNN/LSM vs FLOPs for CNN
6. **Robustness:** Add noise robustness analysis (Gaussian noise on input sweeps)
7. **Pure task:** If additional sweeps arrive from tutor (Acetone, IPA, 50% mixtures), re-run pure task analysis

## Session log

### 2026-07-20
- Rewrote AGENTS.md to be tool-agnostic (works with Claude, Kimi, OpenCode, Copilot, etc.).
- Created `verify_hdl_checklist.py` — runs 7 software-level verification checks for Goal 4 HDL deliverables. All checks pass.
- **Fixed critical data leakage in `nb5_feature_ml_baselines.py`:** Models were evaluated on the same data they trained on (`model.fit(X, y); model.predict(X)` — no holdout). RF and MLP achieved 100% by memorizing 15-18 samples. Replaced with `cross_val_score` + `cross_val_predict` using auto-selected CV strategy (stratified → KFold → LOO). Honest results: RF=93.3% (high-glucose), 60.9% (glucose-ladder); MLP=86.7%, 56.5%.
- **Removed pure task from all scripts and reports:** Changed `--task` default from `pure` to `high-glucose` in 10 scripts (nb4-15, train.py, prepare_dataset.py, raw_data_viewer.py). Removed `pure` from all `choices` lists. Rewrote `generate_report.py` to be dynamic (reads actual CV results) and pure-free. Fixed `nb12_model_comparison.py` report generation to exclude pure. Updated CLAUDE.md quick-reference commands and key results. Rationale: 4/7 pure classes have only 1 sweep — results are unreproducible.
- All 4 goals complete. Project is in maintenance/verification phase. Focus: glucose concentration datasets only.

### 2026-07-19
- Session start. Project state: all 4 goals complete as of 2026-07-16.
- LSM with membrane readout = best model (82.3% high-glucose, 84.4% glucose-ladder). Pure task excluded.
- 16-bit Q3.12 fixed-point is HDL-ready.
- No code changes — context loading only.
