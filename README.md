# PMUT Neuromorphic Classification

Piezoelectric Micromachined Ultrasonic Transducer (PMUT) fluid classification using
spiking neural networks, reservoir computing (ESN/LSM), and CNNs. Impedance sweeps are
recorded across a frequency ramp (100 kHz – 2.5 MHz) with four channels per sweep:
**amplitude**, **phase**, **x_v** (in-phase), **y_v** (quadrature).

**Primary benchmark:** 5-class high-glucose (50/100/250/500/1000 mM), 3 physical sweeps
per class, 3-fold sweep-level rotation with 5 seeds.

**GitHub:** [JImmycheng23/PMUT_SNN](https://github.com/JImmycheng23/PMUT_SNN)
**Branch:** `review-fixes-2026-07-21`

---

## Latest Results (2026-07-25)

| Model | Type | Prediction | Accuracy |
|-------|------|-----------|----------|
| Random Forest | 61 corrected features | sweep | **93.3%** CV |
| LSM (spike readout) | Spiking reservoir | sweep | **88.0% ± 11.8%** |
| ESN (tanh) | Rate-coded reservoir | sweep | **84.0% ± 8.6%** |
| LSM (membrane readout) | Spiking reservoir | sweep | 73.3% ± 9.4% |
| CNN1D | Convolutional | window¹ | 73.8% ± 9.8% |
| SpikingCNN1D | Learned spiking | window¹ | 75.7% (fold 0) |
| Encoder→SNN (rate) | Explicit spike | window | 32.5% (chance=20%) |

¹ Neural models pending sweep-aggregation re-run. Large ± values = genuine across-fold
variance from small dataset (15 sweeps), not leakage.

---

## Quick Start

```sh
# Prepare dataset
python prepare_dataset.py --task high-glucose --split grouped

# Feature extraction (61 features, local-band Q/bw)
python notebooks/nb4_feature_extraction.py --task high-glucose

# Feature ML baselines
python notebooks/nb5_feature_ml_baselines.py --task high-glucose

# Neural models (5 seeds)
python train.py --task high-glucose --split grouped --model cnn1d --num-seeds 5 --epochs 150

# Reservoir models (5 seeds)
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --num-seeds 5
python notebooks/nb11_reservoir.py --task high-glucose --split grouped --num-seeds 5 --lsm

# Central results
python build_central_results.py
```

---

## Project Structure

```
pmut_snn_models/
├── src/pmut_models/       # Core library
│   ├── data.py            # Loading, windowing, 3-fold rotation, phase validation
│   ├── spiking.py         # SNN/CNN models (CNN1D, SpikingCNN1D, EncodedSpikeSNN, …)
│   ├── train_utils.py     # Training loop, multi-seed, sweep aggregation, encoder diagnosis
│   ├── reservoir.py       # ESN + LSM (Liquid State Machine)
│   ├── fixed_point.py     # Q-format, FixedPointLIF, IntegerLSM, HDL test vectors
│   └── config.py          # Default paths
├── notebooks/             # Experiment notebooks
│   ├── nb4_feature_extraction.py    # 61 corrected features (local-band Q/bw)
│   ├── nb5_feature_ml_baselines.py  # RF=93.3%, SVM, LR, KNN, MLP
│   ├── nb8_spike_encoding.py       # Rate/latency/delta/threshold/population encoders
│   ├── nb9_encoder_evaluation.py   # Encoder comparison
│   ├── nb11_reservoir.py           # ESN + LSM training
│   ├── nb12_model_comparison.py    # Cross-model comparison + pipeline selection
│   ├── nb13_fixed_point_lif.py     # Float vs fixed-point LIF comparison
│   ├── nb14_fixed_point_pmut.py    # Retrained-readout reservoir quantisation
│   ├── nb15_hdl_test_vectors.py    # HDL test vector generation
│   └── nb16_hdl_spec.py            # HDL specification generation
├── outputs/               # Generated results (fold-based layout)
│   ├── features/          # NB4: feature table + diagnostic band plots
│   ├── feature_ml/        # NB5: baseline summary + confusion matrices
│   ├── runs/              # Neural models: {task}_folds_w{win}/{model}/fold{N}/
│   ├── reservoir/         # ESN + LSM: {task}/{split}/fold{N}/[lsm/]
│   ├── comparison/        # NB12: comparison tables + pipeline selection
│   ├── fixed_point/       # NB13/NB14: bit-width study
│   ├── hdl_test_vectors/  # NB15: CSV + JSON test vectors
│   ├── hdl/               # NB16: HDL spec + corrected Verilog LIF
│   └── central_results/   # Master CSV (22 structured columns)
├── outputs_legacy/        # Pre-freeze outputs (v2.0-clean-slate)
├── build_central_results.py  # Central results aggregator
├── REPRODUCIBILITY.md     # Frozen commit, data checksums, regeneration commands
└── CLAUDE.md              # AI assistant guidance
```

---

## Evaluation Protocol

**3-fold sweep rotation:** Each fold uses a different physical sweep for train/val/test.

| Fold | Train | Val | Test |
|------|-------|-----|------|
| 0 | Sweep 1 | Sweep 2 | Sweep 3 |
| 1 | Sweep 2 | Sweep 3 | Sweep 1 |
| 2 | Sweep 3 | Sweep 1 | Sweep 2 |

**Multi-seed:** 5 seeds `[7, 42, 123, 999, 2024]` per fold = 15 independent runs per model.

**Sweep-level aggregation:** Window predictions are grouped by sweep_key and combined
via majority vote (or mean logits when probabilities are available). Reservoir results
are sweep-aggregated; neural models pending.

---

## Key Design Decisions

- **glucose-ladder (2–20 mM) removed from primary study.** Single-sweep classes cannot
  support independent train/test. Retained in `data.py` for exploratory visualisation.
- **Phase preprocessing verified.** All checks pass: degrees, 0 wraps in windows, 0 sign
  errors. Phase carries less discriminative info than amplitude for damped PMUT — this
  is a physical property, not a bug.
- **ESN and LSM perform similarly on the primary benchmark.** The previously claimed
  "LSM beats ESN" was based on the now-removed 13-class glucose-ladder.
- **Encoder→SNN ~32% is a fundamental information-loss problem**, not a bug. Binary
  encoding + single channel + fixed projection cannot recover the fine amplitude
  variations that discriminate glucose concentrations.
- **16-bit fixed-point preserves accuracy.** IntegerLSM with quantized readout exists
  as a hardware-ready reference model.

---

## HDL Status

| Deliverable | Status |
|-------------|--------|
| HDL specification | ✅ Complete |
| Verilog LIF neuron (v_next corrected) | ✅ Complete |
| Self-checking testbench | ✅ Built-in sanity test |
| PMUT spike-to-current scaling | ✅ Fixed (binary gate) |
| Test vectors | ⚠️ In `outputs_legacy/` — need regeneration |
| Multi-neuron reservoir | ❌ Not yet |
| Icarus/Verilator simulation | ❌ Not run |

---

## License

MIT
