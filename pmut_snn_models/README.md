# PMUT Neuromorphic Classification

**Piezoelectric Micromachined Ultrasonic Transducer (PMUT)** fluid classification using spiking neural networks (SNN), reservoir computing (ESN/LSM), and conventional CNNs. Classifies glucose concentrations from electrical impedance sweep data.

Each sweep records four signal channels — **amplitude**, **phase**, **x_v** (in-phase/R), **y_v** (quadrature/X) — across a frequency ramp. The frequency axis is treated as a temporal/sequential dimension for spiking and reservoir models.

---

## Project Status

| Goal | Scope | Status |
|---|---|---|
| **Goal 1** | Data pipeline, feature extraction, classical ML baselines | ✅ Complete |
| **Goal 2** | Spike encoding methods, SNN classifiers | ✅ Complete |
| **Goal 3** | Reservoir computing (ESN + LSM), cross-model comparison | ✅ Complete |
| **Goal 4** | Fixed-point quantization, HDL spec + Verilog LIF neuron | ✅ Complete |

**All 4 goals complete (2026-07-16).** Best model: **LSM (LIF reservoir) with membrane readout**, 16-bit Q3.12 fixed-point — HDL-ready.

---

## Quick Start

### Install Dependencies

```powershell
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.2+, NumPy, Pandas, scikit-learn, Matplotlib, SciPy, Seaborn, openpyxl.

### Prepare Data

Place raw data files in C:\Users\27010\Desktop\Neumat\Data\PMUT-SNN\ (or pass --data-root to override):

| File | Contents |
|---|---|
| pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx | Air, Water, Glucose 50–1000 mM (3 sweeps each) |
| low_conc_glucose_sweeps_data_2mM_to_50mM.xlsx | Glucose 2–20 mM + Air/Water (Tx-Rx sheets) |
| Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv | 6 media (1 sweep each, 2000 pts/sweep) |

Prepare datasets:

```powershell
python prepare_dataset.py --task high-glucose --split grouped
python prepare_dataset.py --task glucose-ladder --split window
```

### Run the Full Pipeline

```powershell
# Goal 1: Feature extraction + classical ML
python notebooks/nb4_feature_extraction.py --task high-glucose
python notebooks/nb5_feature_ml_baselines.py --task high-glucose

# Goal 2: Spike encoding + SNN training
python notebooks/nb8_spike_encoding.py --task high-glucose
python notebooks/nb9_encoder_evaluation.py --task high-glucose
python train.py --task high-glucose --split grouped --model cnn1d --epochs 150

# Goal 3: Reservoir computing + model comparison
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --analyze
python notebooks/nb12_model_comparison.py

# Goal 4: Fixed-point + HDL
python notebooks/nb14_fixed_point_pmut.py --task high-glucose --split grouped
python notebooks/nb15_hdl_test_vectors.py --task high-glucose --split grouped
python notebooks/nb16_hdl_spec.py --generate-hdl
```

### View Raw Data

```powershell
python raw_data_viewer.py --task high-glucose
```

Tkinter GUI for browsing sweeps, selecting label, channel, and frequency range.

## Tasks & Splits

### Available Tasks

| Task | Classes | Sweeps | Split | Description |
|---|---|---|---|---|
| high-glucose | 5 | 15 (3/class) | grouped | Glucose 50/100/250/500/1000 mM — primary benchmark |
| glucose-ladder | 13 | 23 | window | Glucose 2mM–1000mM sparse ladder — hardest task |
| pure | — | — | — | Excluded — 4/7 classes have only 1 sweep; results unreproducible |

### Split Strategies

| Split | Description | Best For |
|---|---|---|
| grouped | Hold out last complete sweep per class | Most defensible — requires ≥2 sweeps/class |
| window | Every Nth window (non-overlapping stride) | Tasks with too few sweeps for grouped |
| random_window | Randomly shuffled windows per class | Prevents frequency-range bias |

Default windowing is non-overlapping (stride = window_size, typically 128). Override with --stride 64 for overlapping windows.

## Models

### Neural Classifiers (src/pmut_models/spiking.py)

| Model | Type | Architecture |
|---|---|---|
| CNN1D | Non-spiking baseline | Conv1d + ReLU + AdaptiveAvgPool1d |
| SpikingCNN1D | Spiking CNN | Conv1d + LIF layers over frequency axis |
| TemporalLIFSNN | Spiking RNN-like | LIF neuron per frequency point (freq = time) |
| RateSNNMLP | Spiking MLP | Rate-coded MLP with LIF hidden layers |

### Reservoir Models (src/pmut_models/reservoir.py)

| Model | Dynamics | Update Rule |
|---|---|---|
| Reservoir (ESN) | Rate-coded tanh | h(t) = (1−α)·h(t−1) + α·tanh(W_in·x + W_res·h(t−1)) |
| LiquidStateMachine (LSM) | Spiking LIF | Binary spike events with subtractive reset |

Both reservoirs have fixed random weights — only the ridge regression readout (~300 params) is trained.

### Fixed-Point Models (src/pmut_models/fixed_point.py)

| Model | Purpose |
|---|---|
| FloatLIF | Floating-point gold reference |
| FixedPointLIF | Q-format integer arithmetic — hardware-accurate simulation |

## Key Results

### Best Neural Model: LSM (LIF reservoir, membrane readout)

| Task | LSM (membrane) | CNN1D | ESN (tanh) |
|---|---|---|---|
| high-glucose (5-class, grouped) | 82.3% | 89.8% | 80.4% |
| glucose-ladder (13-class, window) | 84.4% | 81.4% | 75.1% |

LSM dominates on the harder multi-class task (+9.4% over ESN, +3.0% over CNN1D).

### Feature-Based ML Baselines (cross-validated, sweep-level)

| Model | high-glucose (5-class) | glucose-ladder (13-class) |
|---|---|---|
| Random Forest | 93.3% | 60.9% |
| MLP | 86.7% | 56.5% |
| SVM (Linear) | 80.0% | 26.1% |

⚠️ Feature-based ML uses sweep-level engineered features and should not be directly compared to window-level neural models.

### Fixed-Point Quantization (LSM pipeline)

| Format | high-glucose | glucose-ladder | Hardware |
|---|---|---|---|
| Float (reference) | 84.8% | 85.0% | — |
| 16-bit Q3.12 | 86.1% | 88.0% | ✅ FPGA DSP48 |
| 12-bit Q3.8 | 83.0% | 86.2% | ⚠️ Viable |
| 8-bit Q3.4 | 77.8% | 77.8% | ❌ Marginal |

16-bit fixed-point improves accuracy (quantization noise acts as implicit regularization on small datasets).

### Best Spike Encoder

Latency coding — 100% accuracy with ~6× fewer spikes than rate coding.

## Project Structure

```text
pmut_snn_models/
├── src/pmut_models/           # Core library
│   ├── config.py              # Default paths
│   ├── data.py                # Data loading, windowing, splits, standardization
│   ├── spiking.py             # SpikeFunction, LIFStateLayer, 4 trainable SNN/CNN models
│   ├── train_utils.py         # Training loop, evaluation, multi-seed harness
│   ├── reservoir.py           # ESN (tanh), LSM (LIF), grid search, memory capacity
│   └── fixed_point.py         # Q-format, FloatLIF, FixedPointLIF, HDL test vectors
├── notebooks/                 # Analysis & experiment scripts (NB4–NB16)
│   ├── nb4_feature_extraction.py      # Extract 12 physical features per sweep
│   ├── nb5_feature_ml_baselines.py    # Classical ML with honest cross-validation
│   ├── nb7_lif_validation.py          # LIF neuron dynamics validation
│   ├── nb8_spike_encoding.py          # 5 spike encoding methods
│   ├── nb9_encoder_evaluation.py      # Encoder comparison with LR classifier
│   ├── nb11_reservoir.py              # ESN + LSM training, analysis, sweeps
│   ├── nb12_model_comparison.py       # Cross-model comparison + Goal 3 report
│   ├── nb13_fixed_point_lif.py        # Fixed-point LIF neuron validation
│   ├── nb14_fixed_point_pmut.py       # Quantized PMUT inference bit-width study
│   ├── nb15_hdl_test_vectors.py       # HDL test-vector generation from PMUT data
│   ├── nb16_hdl_spec.py               # HDL specification + Verilog LIF neuron
│   └── generate_report.py              # Goal 1+2 completion report generator
├── train.py                   # Neural model training CLI
├── prepare_dataset.py         # Dataset preparation (.npz generation)
├── raw_data_viewer.py         # Raw sweep visualization (Tkinter GUI)
├── verify_hdl_checklist.py    # Software-level HDL verification (7 checks)
├── outputs/                   # All generated artifacts
│   ├── datasets/              # Prepared .npz files
│   ├── features/              # Extracted PMUT features (NB4)
│   ├── feature_ml/            # Classical ML baselines (NB5)
│   ├── runs/                  # Neural model training runs
│   ├── lif_validation/        # LIF neuron experiments (NB7)
│   ├── spike_encoding/        # Spike rasters + stats (NB8)
│   ├── encoder_evaluation/    # Encoder comparison (NB9)
│   ├── reservoir/             # ESN + LSM results (NB11)
│   ├── comparison/            # Cross-model comparison + Goal 3 report (NB12)
│   ├── fixed_point/           # Fixed-point LIF + PMUT study (NB13/NB14)
│   ├── hdl_test_vectors/      # HDL verification data (NB15)
│   ├── hdl/                   # HDL spec + Verilog (NB16)
│   ├── GOAL_1_2_COMPLETION_REPORT.md
│   ├── GOAL_3_COMPLETION_REPORT.md
│   └── GOAL_4_COMPLETION_REPORT.md
├── CLAUDE.md                  # AI coding assistant guide
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Output Reports

After running the pipeline, see the completion reports for detailed results:

| Report | Description |
|---|---|
| outputs/GOAL_1_2_COMPLETION_REPORT.md | Data + Classical ML + SNN |
| outputs/GOAL_3_COMPLETION_REPORT.md | Reservoir computing + model comparison |
| outputs/GOAL_4_COMPLETION_REPORT.md | Fixed-point + HDL |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Frequency as time | PMUT sweeps are inherently sequential. LIF neurons and reservoirs treat frequency steps as a temporal sequence. |
| Non-overlapping windows | Default stride = window_size prevents data leakage between train/test windows. |
| Grouped split is most defensible | Holds out complete sweeps rather than windows from the same sweep. |
| Fixed reservoir weights | Only the readout is trained (ridge regression), making reservoirs extremely data-efficient (important with only 15–23 sweeps). |
| 16-bit Q3.12 | Matches FPGA DSP48 blocks natively; quantization noise provides beneficial regularization. |

## Hardware Path

The LSM → fixed-point → HDL pipeline is ready for synthesis:

```text
PMUT raw data → spike encode → LSM reservoir (LIF) → ridge readout
                                                        ↓
                                             16-bit Q3.12 quantize
                                                        ↓
                                        Verilog LIF neuron (lif_neuron_fxp.v)
                                                        ↓
                                            FPGA synthesis (pending)
```

HDL deliverables in outputs/hdl/:

| Deliverable | Description |
|---|---|
| HDL_SPECIFICATION.md | Complete port map, Q-format, update rule, testbench requirements |
| lif_neuron_fxp.v | Synthesis-ready Verilog module + testbench |

Resource estimate per LIF neuron (16-bit): 1 signed multiplier (16×16), 1 adder, 1 comparator, ~33 flip-flops.

## Data Limitations

| Limitation | Details |
|---|---|
| Small dataset | 15 sweeps (high-glucose) and 23 sweeps (glucose-ladder). Reservoir computing is well-suited to this regime. |
| Pure task excluded | 4 of 7 classes in the pure-fluid dataset have only 1 sweep. Awaiting additional sweeps from the lab. |
| No test-set holdout | Cross-validation is used throughout. A true held-out test set would require additional independent measurements. |
| Feature vs. window level | Feature-based ML (sweep-level features) and neural/reservoir models (window-level raw data) operate on different representations and are not directly comparable. |