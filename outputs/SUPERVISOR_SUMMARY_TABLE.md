# PMUT Neuromorphic Project — Response to Supervisor Review

**Date:** 2026-07-21  
**Branch:** `review-fixes-2026-07-21`  
**Repository:** https://github.com/JImmycheng23/PMUT_SNN

---

## Point-by-Point Summary

| Point | Your explanation | Is a change required? | What you changed | New output/result location |
|---|---|---|---|---|
| **1** | Confirmed: primary tasks are high-glucose (5-class, 3 sweeps/class) and glucose-ladder (13-class, 23 sweeps). Sweep inventory table provided. Pure task excluded. | Yes — final paper should remove IPA/acetone | Sweep inventory table created | Sweep inventory in full response |
| **2** | Global `argmax(amplitude)` was picking up sweep boundary artifacts (100 kHz / 2.5 MHz). PMUT is heavily overdamped (Q_eff < 0.5). | ✅ Fixed 2026-07-21 | 4-band hybrid multi-peak features + spectral shape features in NB4 | `outputs/features/` |
| **3** | 100% values were training-set accuracy (`fit+predict` on same data). Fixed with `cross_val_predict`. Then scaler leakage found (Point 12). | ✅ Fixed 2026-07-20 + 2026-07-21 | NB5: `cross_val_predict` then `make_pipeline(StandardScaler(), model)` | `outputs/feature_ml/` — RF 93.3% (high-glucose), 56.5% (glucose-ladder) |
| **4** | NB9 trained and tested LR on same sweeps — 100% was training accuracy. | ✅ Fixed 2026-07-21 | `split_indices()` for sweep-level train/test; honest results: delta 40% test acc | `outputs/encoder_evaluation/` |
| **5** | Two pipelines were disconnected. `train.py` never imported NB8 encoders. | ✅ Fixed 2026-07-21 | Built `EncodedSpikeSNN` model + `encode_dataset_bundle()` + `--encode` flag. Pipeline now connected. | `src/pmut_models/spiking.py`, `train_utils.py`, `train.py` |
| **6** | Zero saved outputs existed for any SNN/CNN model. Numbers in report not backed by files. | ✅ Regenerated 2026-07-21 | All models rerun with honest train/val/test; saved to `outputs/runs/` | `outputs/central_results/master_results.csv` |
| **7** | Grouped split holds out last sweep per class. Confirmed correct. | No | N/A — already correct | `data.py:247-261` |
| **8** | 8/13 classes have only 1 sweep. Window split is within-sweep evaluation, not independent test. | Language only | Documented honestly — this limitation is clearly stated | §Point 8 in full response |
| **9** | Low-conc (100k–1.5M) vs high-conc (100k–2.5M) — different ranges created acquisition confound. | ✅ Fixed 2026-07-21 | `data.py`: filter high-conc to ≤1.5 MHz. No resonant info lost (all modes ≤1.2 MHz). | `data.py:186-188` |
| **10** | LSM 82.3%/84.4% were hardcoded in NB12, never produced by any experiment. LSM class existed but NB11 never called it. | ✅ Fixed 2026-07-21 | Added `train_lsm()` to NB11; ran 3-seed LSM on both tasks; NB12 now loads dynamically | `outputs/reservoir/*/lsm/metrics_summary.json` |
| **11** | "amplitude_only" used [A, x_v], "phase_only" used [φ, y_v] — x_v/y_v mix information. | ✅ Fixed 2026-07-21 | Clean channels: [A], [φ], [A+φ], [x_v+y_v]. Phase alone = 43.5% (was 71.7%). | `outputs/reservoir/*/modality_comparison.json` |
| **12** | `StandardScaler.fit_transform(X)` before CV leaked test-fold statistics. Inflated LR by 13 pp. | ✅ Fixed 2026-07-21 | `make_pipeline(StandardScaler(), model)` — scaler fitted per CV fold | `nb5_feature_ml_baselines.py` |
| **13** | Test set used for epoch selection and final eval — no validation set. | ✅ Fixed 2026-07-21 | `DatasetBundle.x_val/y_val`; train/val/test split (368/92/230); val for epoch selection, test once | `data.py`, `train_utils.py`, `train.py` |
| **14** | `reservoir_grid_search()` received `x_test, y_test` — HP selected by test accuracy. | ✅ Fixed 2026-07-21 | Internal `train_test_split` on training data; ranked by val_acc, test never touched | `reservoir.py`, `nb11_reservoir.py` |
| **15** | "Temporal dynamics" language misleading — frequency bins are not physical time. | ✅ Fixed 2026-07-21 | Changed to "sequential spectral processing" across 6 files | `CLAUDE.md`, `README.md`, `reservoir.py`, etc. |
| **16** | "16-bit regularization" claim untested. Readout is retrained at each bit width. | ✅ Fixed 2026-07-21 | Documented as "retrained-readout quantization"; regularization claims removed | `CLAUDE.md`, `README.md`, `nb14_fixed_point_pmut.py` |
| **17** | Only overall accuracy reported — misleading for imbalanced 13-class task. | ✅ Fixed 2026-07-21 | `balanced_accuracy`, `macro_f1`, `per_class_recall`, confusion matrix in every `metrics.json` | `train_utils.py`, `nb11_reservoir.py` |
| **18** | All changes should be preserved with clear versioning. | ✅ Done | Git repo initialized; branch `review-fixes-2026-07-21` pushed to GitHub | https://github.com/JImmycheng23/PMUT_SNN/tree/review-fixes-2026-07-21 |

---

## Key Honest Results (after all fixes)

### Neural Models — high-glucose grouped (train/val/test split, seed 7)

| Model | Best Val Acc | Final Test Acc |
|---|---|---|
| CNN1D (non-spiking) | 88.0% | **84.3%** |
| SpikingCNN1D | 87.0% | 83.0% |
| RateSNNMLP | 85.9% | 73.5% |
| TemporalLIFSNN | 83.7% | 76.1% |

### Reservoir Models — 3 seeds

| Task | ESN (tanh) | LSM (membrane) |
|---|---|---|
| high-glucose grouped | 80.9% ± 0.4% | 80.3% ± 1.4% |
| glucose-ladder window | 70.1% ± 1.1% | **81.4% ± 1.2%** |

### Feature ML (CV, Pipeline, no scaler leakage)

| Model | high-glucose | glucose-ladder |
|---|---|---|
| Random Forest | **93.3%** | **56.5%** |
| MLP | 80.0% | 43.5% |

### Clean Modality Ablation (ESN, high-glucose)

| Modality | Accuracy |
|---|---|
| Amplitude only | 73.5% |
| Phase only | 43.5% |
| Amplitude + Phase | 81.7% |
| Complex I/Q | **82.2%** |

---

**Central results:** `outputs/central_results/master_results.csv` — 26 rows, machine-readable  
**Full response:** `outputs/RESPONSE_TO_SUPERVISOR_REVIEW.md` — detailed per-point analysis with code references  
**Repository:** https://github.com/JImmycheng23/PMUT_SNN — branch `review-fixes-2026-07-21`
