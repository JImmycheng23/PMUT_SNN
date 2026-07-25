# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Kimi, OpenCode, GitHub Copilot, Cursor, etc.) when working with code in this repository.

## Project overview

PMUT (Piezoelectric Micromachined Ultrasonic Transducer) neuromorphic classification project. Classifies fluids from electrical impedance sweep data using spiking neural networks (SNN), reservoir computing (ESN/LSM), and conventional CNNs. Each sweep records four signal channels — `amplitude`, `phase`, `x_v` (in-phase/R), `y_v` (quadrature/X) — across a frequency ramp. The frequency axis is treated as a sequential spectral dimension (ordered from low to high frequency — NOT physical time).

**Current best result:** ESN (tanh reservoir) = **80.9%** and LSM (LIF reservoir, membrane readout) = **80.3%** on 5-class high-glucose (grouped). ESN and LSM achieve broadly similar performance on the primary benchmark (difference within seed variance). No model “beats” the others — different architectures trade off stability (ESN has lower variance) vs biological plausibility (LSM uses spike-based computation). 16-bit fixed-point quantization preserves accuracy (observed improvements at 16-bit are within seed variance — readout is retrained at each bit width, see Point 16). Pipeline is HDL-ready. Pure task excluded (4 of 7 classes have only 1 sweep).

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

# Extract 4-band hybrid multi-peak + spectral shape features
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
python notebooks/nb9_encoder_evaluation.py --task high-glucose --split grouped
python notebooks/nb9_encoder_evaluation.py --task glucose-ladder --split window

# Encoder → SNN pipeline (explicit encoding + spiking classifier)
python train.py --task high-glucose --split grouped --model encoded_spike_snn --encode rate --epochs 150
python train.py --task high-glucose --split grouped --model encoded_spike_snn --encode latency --epochs 150

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

### Pipeline: PMUT signal → spike encoder → SNN

This project implements a two-stage spiking pipeline:

```
PMUT sweep → explicit spike encoder (rate/latency/delta/etc.) → spike trains → SNN → classification
                                                                              ↑
                                                                    EncodedSpikeSNN model
```

1. **Explicit encoder** (NB8 / NB9): Converts continuous amplitude to binary spike trains using a hand-designed encoding function. Compared 5 strategies; rate coding preserves the most information for downstream learning.
2. **Spiking classifier** (`EncodedSpikeSNN` in `spiking.py`): Takes the encoded spike trains and processes them through trainable LIF neurons. The connection is wired via `train.py --encode <encoder> --model encoded_spike_snn`.

The earlier models (`TemporalLIFSNN`, `SpikingCNN1D`, etc.) take **continuous-valued** spectra as input and learn their own internal spike representations — they bypass the explicit encoder. Those models achieve higher accuracy (CNN1D = 89.8%) because they use all 4 signal channels and learn the projection. The encoder→SNN pipeline uses only the amplitude channel and a fixed encoding, trading accuracy for a hardware-realistic neuromorphic front-end.

**Encoder → SNN results (high-glucose, grouped, win=128, 150 epochs):**

| Pipeline | Best Test Acc | Notes |
|---|---|---|
| Rate encoder → EncodedSpikeSNN | **32.5% ± 1.1%** | 100% train acc (overfitting); 50 time steps × 128 features |
| Latency encoder → EncodedSpikeSNN | 31.7% ± 1.3% | ~6× fewer spikes than rate |
| Delta encoder → EncodedSpikeSNN | 28.3% ± 0.4% | Too few spikes (sensitivity=1.5 on damped signals) |
| _CNN1D (continuous, 4-channel)_ | _89.8%_ | Reference: learned projection, all channels |
| _Chance_ | _20%_ | 5-class baseline |

**Key finding:** The explicit-encoder pipeline is functional and beats chance by 12.5 percentage points, but information loss from (a) binary encoding, (b) using only amplitude, and (c) hand-designed vs learned projection creates a ~3× gap compared to continuous-input models. Rate coding provides the best accuracy-efficiency trade-off for this pipeline.

### Models

#### Trainable neural models (`spiking.py`)

| Model | Type | Input → Processing |
|---|---|---|
| **CNN1D** | Non-spiking baseline | Continuous [A,φ,x_v,y_v] → Conv1d + ReLU |
| **SpikingCNN1D** | Learned spiking | Continuous [A,φ,x_v,y_v] → Conv1d → LIF |
| **TemporalLIFSNN** | Learned spiking | Continuous [A,φ,x_v,y_v] → Linear per freq step → LIF recurrence |
| **RateSNNMLP** | Learned spiking | Continuous [A,φ,x_v,y_v] → Flatten + Linear → LIF rate-coded over virtual time |
| **EncodedSpikeSNN** | Explicit-encoder pipeline | **Spike trains** from NB8 encoder → Linear per time step → LIF → class |

The first four models take raw continuous spectra. Only `EncodedSpikeSNN` uses pre-encoded spike trains. `RateSNNMLP`'s "rate" refers to virtual time steps, not the NB8 `rate_encode()` function.

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
| NB4 | `nb4_feature_extraction.py` | 1 | `outputs/features/{task}_feature_table.csv` — 4-band hybrid multi-peak + spectral shape features, `_*.png` |
| NB5 | `nb5_feature_ml_baselines.py` | 1 | `outputs/feature_ml/{task}_*.csv`, `_*.png` |
| NB7 | `nb7_lif_validation.py` | 2 | `outputs/lif_validation/exp{1-4}_*.png` |
| NB8 | `nb8_spike_encoding.py` | 2 | `outputs/spike_encoding/{task}_spike_rasters.png`, `_stats.csv` |
| NB9 | `nb9_encoder_evaluation.py` | 2 | `outputs/encoder_evaluation/{task}_*.csv`, `_*.png` — sweep-level train/test split via `split_indices()` |
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

**Feature-based ML (cross-validated, sweep-level spectral-shape features, NOT comparable to window-level neural models):**

⚠️ **Important (2026-07-24 — corrected):** The PMUT in liquid-loaded Tx-Rx mode is NOT heavily overdamped (Q_eff < 0.5 claim was an artifact — see below). The corrected local-band analysis (NB4, 2026-07-24) finds physically reasonable Q values: mode 2 Q = 63–176, mode 3 Q = 152–2075 across glucose concentrations. The PMUT in liquid is damped but classical resonance metrics are measurable with proper local-band analysis. NB4 now extracts:
- 4-band local-band resonance features (local baseline removal, detrended half-power, crossings restricted to band, linear interpolation, NaN when no clean resonance)
- Fallback features per band: band energy, spectral centroid, local amplitude slope, phase gradient, peak prominence
- Full-band spectral shape features (centroid, variance, skewness, kurtosis, roll-off, slopes)
- I/Q trajectory features (ellipse area, mean radius, phase spread)
- Level features (mean, std, peak, AUC, noise floor)
- Diagnostic band plots showing peak markers, half-power crossings, and band boundaries per sweep

The old RF results (93.3%, 60.9%) used spurious global-argmax features. Results below use honest multi-peak + spectral shape features.

*high-glucose (5-class, 3-fold stratified CV, 15 sweeps, 61 features):*
| Model | CV Accuracy |
|---|---|
| Random Forest | 100.0% |
| Logistic Regression | 80.0% |
| MLP | 80.0% |
| SVM (Linear) | 80.0% |

*glucose-ladder (13-class, leave-one-out CV, 23 sweeps, 61 features):*
| Model | CV Accuracy |
|---|---|
| Random Forest | 52.2% |
| Logistic Regression | 47.8% |
| SVM (Linear) | 47.8% |
| MLP | 47.8% |

Top RF features for high-glucose: f0_b2_hz (0.45 MHz mode frequency), amp_mean, q_b2, amp_b3 (0.9 MHz mode amplitude) — physically meaningful multi-peak features. 15 sweeps with 61 features risks overfitting; RF's 100% on high-glucose may reflect this — the primary results remain window-level neural models.

⚠️ Previous reports of 100% accuracy for RF/MLP were caused by a data-leakage bug (training and testing on identical samples in nb5, lines 163-165). Fixed 2026-07-20 — metrics now use `cross_val_predict`. Additionally, pre-2026-07-21 NB4 used `argmax(amplitude)` which picked up sweep boundaries as spurious "resonance" features.

⚠️ Pure task excluded — 4 of 7 classes have only 1 sweep, making even LOO CV unreliable (singleton classes have zero training examples in their test fold).

**Explicit spike encoder → SNN pipeline:** Rate coding provides the best encoder→SNN accuracy at **32.5% ± 1.1%** (high-glucose, grouped, chance=20%). The full pipeline PMUT → explicit encoder → spike trains → EncodedSpikeSNN is functional and connected via `train.py --encode`. The ~3× gap vs continuous-input CNN1D (89.8%) reflects: (a) binary encoding information loss, (b) using only amplitude (vs 4 channels), and (c) hand-designed vs learned projection. Latency coding is close behind (31.7%) with ~6× fewer spikes. For NB8/NB9 encoder comparison (LR on aggregate features), delta coding shows best test generalisation on high-glucose (40% vs chance=20%), but the LR+aggregate-features approach is fundamentally different from the encoder→SNN pipeline.

### Goal 3: Reservoir computing + Model comparison (2026-07-14)

**ESN vs LSM head-to-head (3 seeds, mean ± std):**

| Task | ESN (tanh) | LSM (membrane) | LSM (spikes) | Δ (LSM−ESN) |
|---|---|---|---|---|
| high-glucose (5-class, grouped) | 80.43% ± 0.61% | 82.32% ± 2.36% | **82.46%** ± 2.36% | **+1.89%** |
| glucose-ladder (13-class, window) | 75.05% ± 0.56% | **84.43%** ± 0.49% | 81.64% ± 0.28% | **+9.38%** |

**ESN and LSM perform similarly on the primary 5-class benchmark** (ESN 80.9% ± 0.4%, LSM membrane 80.3% ± 1.4% — difference within seed variance). ESN has lower variance; LSM offers spike-based computation for neuromorphic hardware. On the now-removed 13-class glucose-ladder, LSM showed stronger advantage (+11 pp over ESN), but that task used within-sweep evaluation on single-sweep classes and is no longer a primary result.

**Why LSM > ESN:**
1. **Sparse coding** — LIF neurons fire only on threshold crossing; tanh neurons are always active
2. **Event-driven recurrence** — binary spike feedback is all-or-nothing; continuous feedback blurs over time
3. **Reset dynamics** — subtractive reset prevents any single neuron from dominating the reservoir

**Input modality ablation (reservoir):** Amplitude alone often beats amplitude+phase. Phase adds noise for concentration-based classification tasks.

**Pipeline selected for hardware:** LSM (LIF, membrane readout) — spiking neurons map directly to neuromorphic hardware. ESN (tanh) provides equivalent accuracy with lower variance on the primary benchmark.

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

**Key finding:** 16-bit fixed-point matches or slightly exceeds float accuracy on both tasks. The observed +1–3% at 16-bit is within seed variance and may reflect the retrained-readout procedure (a new Ridge readout is fitted for each bit width — see Point 16). The claim of "implicit regularization" from quantization noise has not been directly tested and should be treated as a hypothesis, not a conclusion.

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
- Feature-based ML uses sweep-level spectral shape features (20 spectral/shape features + 12 multi-peak features = 32 total) and should not be directly compared to window-level neural models. RF achieves 100.0% CV on high-glucose (caution: 15 sweeps × 61 features risks overfitting), 52.2% on glucose-ladder. The PMUT in liquid is heavily overdamped — classical resonance metrics (f₀, BW, Q) are not used; instead 4-band hybrid multi-peak features track individual PMUT modes. Previous 100% claims (2026-07-20) were caused by train-on-test leakage; previous f₀/BW/Q values were boundary artifacts from global argmax (fixed 2026-07-21).

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

### 2026-07-25
- **Fixed Point 7 (review):** Completed comprehensive encoder→SNN diagnosis covering all 7 sub-points:
  [1] All five NB8 encoders tested (rate/latency/threshold/delta/population).
  [2] Amplitude-only vs amplitude+phase compared per encoder.
  [3] Per-sample sparsity measured: rate=14.8% dense, latency=2.0% (too sparse), delta=86% silent steps.
  [4] Timing preservation: rate shuffle_corr=0.01 (frequency ordering DESTROYED by Poisson noise); latency autocorr=0.90 (preserves structure but too sparse).
  [5] Full tensor dimension pipeline traced: (n_windows, n_steps, n_features) verified at every stage.
  [6] Confirmed SNN receives binary spike TRAINS (not statistical aggregates).
  [7] Information content: MI≈0, LR=25% (chance) for ALL encoders — mean spike count carries zero class-discriminative information.
  **Tested all-4-channel encoding (A,φ,x_v,y_v → 512 features):** test accuracy 30.9% vs 25.2% amp-only — marginal gain, severe overfitting (100% train by epoch 30). The bottleneck is binary encoding, not channel count.
  **Final verdict:** The ~3× gap vs CNN1D (~84%) is a FUNDAMENTAL information-loss problem. Binary spike conversion + hand-designed encoding + single LIF layer cannot recover the fine amplitude variations (<5%) that discriminate glucose concentrations. This is an honest negative result — the explicit-encoder pipeline is functional (beats chance by 10–12 pp) but is not competitive with continuous-input models. Documented in CLAUDE.md and RESPONSE_TO_SUPERVISOR_REVIEW.md.

### 2026-07-24
- **Fixed Point 1 (review):** Removed low-concentration ladder (2–20 mM) from primary classification. The 5-class high-glucose (50–1000 mM) is now the sole primary benchmark. glucose-ladder retained in data.py for exploratory visualisation only with prominent warning docstring. Updated --task choices in 11 scripts.
- **Fixed Point 2 (review):** Clean freeze. All legacy outputs moved to outputs_legacy/. Fresh outputs/ for regeneration. Created REPRODUCIBILITY.md with frozen commit hash (5f58f06, tag v2.0-clean-slate), data file MD5 checksums, and exact regeneration commands.
- **Fixed Point 3 (review):** Corrected PMUT bandwidth/Q extraction in NB4. The previous Q < 0.5 values were artifacts of searching half-power crossings on raw amplitude with default full-sweep fallback (left=0, right=n-1). New local-band method: (1) local baseline removal within each mode band, (2) peak detection on detrended signal via prominence, (3) half-power computed on detrended signal, (4) crossing search restricted to local band with linear interpolation, (5) NaN sentinel when no clean resonance. Added fallback features per band (energy, centroid, slope, phase gradient, prominence), full-band spectral shape features (centroid, variance, skewness, kurtosis, roll-off, slopes), I/Q trajectory features (ellipse area, mean radius, phase spread), and diagnostic band plots. Honest Q values: mode 2 Q = 63–176, mode 3 Q = 152–2075. Feature count increased from 32 to 61. The previously claimed "heavily overdamped" conclusion was wrong — it was an artifact of the extraction method.

### 2026-07-21 (continued)
- **Built encoder→SNN pipeline (connection was missing):** The assignment intended PMUT → encoder → SNN, but the codebase had two disconnected pipelines. Added `EncodedSpikeSNN` model to `spiking.py` (processes pre-encoded spike trains through LIF neurons), `encode_dataset_bundle()` to `train_utils.py` (wraps NB8 encoders around DatasetBundle), and `--encode` flag to `train.py`. The pipeline is now functional: `python train.py --model encoded_spike_snn --encode rate`. Results on high-glucose (grouped, win=128, 150ep, 3 seeds): rate→SNN **32.5% ± 1.1%**, latency→SNN **31.7% ± 1.3%**, delta→SNN 28.3% ± 0.4% (chance=20%). The ~3× gap vs continuous CNN1D (89.8%) is the cost of explicit encoding + single-channel input. Updated CLAUDE.md pipeline section, common commands, and key results.

### 2026-07-21
- **Fixed critical data-leakage bug in NB9 (spike encoder evaluation):** Same class of bug as NB5 — `fit(X, y); predict(X)` on identical data. Rewrote to use `split_indices()` for physical-sweep-level train/test separation. Honest results: high-glucose delta 40%, latency 20%; glucose-ladder rate/delta 15.4%. Updated GOAL_1_2_COMPLETION_REPORT.md NB9 section.
- **Clarified two distinct spiking approaches in documentation:** The project has TWO independent spiking pipelines, not a single "PMUT → encoder → SNN" chain. **(A) Explicit spike encoding** (NB8/NB9): PMUT amp → hand-designed encoder (rate/latency/delta/etc.) → spike trains → aggregate features → LR. Answers "what encoding preserves class info?" for neuromorphic front-end design. **(B) Learned spiking network** (train.py, spiking.py, reservoir.py): continuous PMUT [A,φ,x_v,y_v] → trainable projection → LIF dynamics (internal spikes) → classification. Answers "can a spiking network classify PMUT sweeps end-to-end?" The two are completely independent — `train.py` does not import NB8 encoders. Updated CLAUDE.md Models section and GOAL_1_2_COMPLETION_REPORT.md with comparison table. The earlier characterization of an encoder "feeding into" the SNN was incorrect.
- **Fixed critical resonance extraction bug in NB4:** The global `argmax(amplitude)` was picking up sweep boundary artifacts (100 kHz or 2.5 MHz) rather than genuine PMUT resonances. Investigation revealed the PMUT in liquid-loaded Tx-Rx mode is moderately damped: amplitude varies 2-5%, phase varies 4-9° across the full sweep, y_v (quadrature/X) never crosses zero for most concentrations. The initial Q_eff < 0.5 finding was later determined (2026-07-24) to be an artifact of buggy bandwidth extraction (half-power crossings defaulting to full sweep width). Corrected local-band analysis yields physically reasonable Q values (mode 2: 63–176, mode 3: 152–2075).
- **Replaced spurious features with spectral shape + banded multi-peak features:** NB4 now extracts 51 features including:
  - 4-band hybrid multi-peak features (0.2, 0.45, 0.9, 1.2 MHz PMUT modes)
  - Phase-based detection for the deep damped 0.2 MHz mode; amplitude-based detection for higher-Q modes at 0.45-1.2 MHz
  - Spectral shape features (centroid, variance, skewness, kurtosis, slopes, I/Q metrics)
  - Level features (mean, std, AUC, noise floor)
- **Recomputed NB5 feature-based ML baselines:** High-glucose RF=100.0% (up from 93.3% — multi-peak features add discriminative power). Glucose-ladder RF=52.2% (up from 47.8%). Logistic Regression jumps from 21.7% to 47.8% on glucose-ladder. Top features: f0_b2_hz (0.45 MHz mode), amp_mean, q_b2 — physically meaningful.
- Neural models (CNN, LSM, ESN) unaffected — they operate on raw windowed spectra, not extracted features.
- Updated CLAUDE.md key results and caveats.

### 2026-07-20
- Rewrote AGENTS.md to be tool-agnostic (works with Claude, Kimi, OpenCode, Copilot, etc.).
- Created `verify_hdl_checklist.py` — runs 7 software-level verification checks for Goal 4 HDL deliverables. All checks pass.
- **Fixed critical data leakage in `nb5_feature_ml_baselines.py`:** Models were evaluated on the same data they trained on (`model.fit(X, y); model.predict(X)` — no holdout). RF and MLP achieved 100% by memorizing 15-18 samples. Replaced with `cross_val_score` + `cross_val_predict` using auto-selected CV strategy (stratified → KFold → LOO). Honest results: RF=93.3% (high-glucose), 60.9% (glucose-ladder); MLP=86.7%, 56.5%.
- **Removed pure task from all scripts and reports:** Changed `--task` default from `pure` to `high-glucose` in 10 scripts (nb4-15, train.py, prepare_dataset.py, raw_data_viewer.py). Removed `pure` from all `choices` lists. Rewrote `generate_report.py` to be dynamic (reads actual CV results) and pure-free. Fixed `nb12_model_comparison.py` report generation to exclude pure. Updated CLAUDE.md quick-reference commands and key results. Rationale: 4/7 pure classes have only 1 sweep — results are unreproducible.
- All 4 goals complete. Project is in maintenance/verification phase. Focus: glucose concentration datasets only.

### 2026-07-19
- Session start. Project state: all 4 goals complete as of 2026-07-16.
- ESN and LSM perform similarly on 5-class high-glucose (ESN 80.9%, LSM 80.3%). LSM advantage was previously overstated due to hardcoded numbers; actual saved metrics show them tied within seed variance. Pure task excluded.
- 16-bit Q3.12 fixed-point is HDL-ready.
- No code changes — context loading only.
