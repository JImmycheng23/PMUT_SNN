# Response to Supervisor Review (Fix-point 2 — 15 Points)

**Date:** 2026-07-24
**Repository:** `pmut_snn_models` (local only — GitHub remote disconnected)
**Branch:** `review-fixes-2026-07-21`

---

## Point 1 — Remove low-concentration ladder from main classification

### Status: ✅ ACCEPTED — will implement

**Agreed.** The 2–20 mM low-concentration data has only 1 physical sweep per concentration — it cannot support independent train/test at the sweep level. The current 13-class experiments divide windows from the same physical sweep between train and test, which can demonstrate within-sweep spectral separability but not generalisation to new acquisitions.

Additionally, merging the low (100 kHz–1.5 MHz) and high (100 kHz–2.5 MHz) campaigns introduces an acquisition-campaign confound — even with the current 1.5 MHz frequency clipping.

**Action:**
- Remove `glucose-ladder` as a primary benchmark task
- Main quantitative study: **5-class high-glucose (50–1000 mM) only**, with strict sweep-held-out evaluation
- Low-concentration data retained for exploratory use: spectral plots, concentration-trend visualization, PCA, reservoir-state visualization
- Keep `load_low_glucose()` in `data.py` but add prominent docstring noting it is for visualization only

**One question for you:** Should we also reintroduce Air and DI Water as additional classes to the high-concentration study? The high-concentration file (`pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx`) has 3 sweeps each for Air and DI Water. This would make it a 7-class study (Air, DI Water, Glucose 50/100/250/500/1000 mM). The `load_high_air_di()` and `load_high_glucose()` functions already exist — it's just a matter of task definition.

---

## Point 2 — Repository outputs from different code versions

### Status: ACKNOWLEDGED — will fix with clean freeze

**You're right.** The current `outputs/` directory contains a mixture of:
- Legacy outputs (pre-July-22 fixes)
- Intermediate outputs (partially regenerated)
- Final outputs (after train/val/test split was added)

It is not possible to determine which exact implementation produced each number.

**Action:**
- Move all current `outputs/` → `outputs_legacy/`
- Freeze one clean commit on `review-fixes-2026-07-21`, tag `v2.0-clean-slate`
- Create `REPRODUCIBILITY.md` with commit hash, Python version, dependency versions, data file checksums
- Create `regenerate_all.py` that orchestrates full regeneration from the frozen commit
- Every output file will embed git commit hash in its metadata
- Every result in the central CSV will trace to a specific output file and commit

---

## Point 3 — PMUT resonance, bandwidth, and Q extraction

### Status: ACCEPTED — the current method is flawed

**You identified the exact bug.** After codebase exploration, I can confirm the specific mechanism:

1. **Peak found on detrended signal, half-power computed on raw amplitude.** Peaks are detected on `amp_det` (after degree-2 polynomial detrend). But the -3dB threshold `amp[p] / sqrt(2)` uses RAW `amp`. When the polynomial baseline at low frequencies exceeds the peak amplitude, the raw amplitude never crosses below half-power.

2. **Default values silently produce garbage.** The left/right crossing search defaults to `left=0`, `right=n-1` when no crossing is found. This produces `bw = full_sweep_span` (e.g., 2.4 MHz) and Q values < 0.5. These are artifacts of the extraction method, not evidence of physical overdamping.

3. **Phase-based peaks (band b1) get Q=0 hardcoded** — no bandwidth measurement at all.

4. **No interpolation at crossing points** — bandwidth overestimated by up to one frequency bin (~10–20 kHz) per side.

5. **Crossing search spans full sweep, not local band** — if a neighboring larger peak exists, the search may find that peak's trough.

6. **CLAUME.md claims "spectral shape features" (centroid, variance, skewness, kurtosis, slopes) exist — they don't.** Only mean, peak, and AUC are computed.

**Action — rewrite the extraction per mode band:**
1. Define local frequency band per mode (b1: 120–380, b2: 380–700, b3: 700–1100, b4: 1100–2450 kHz)
2. Estimate and remove local baseline within each band
3. Detect peak on detrended local signal using prominence
4. Compute half-power on detrended signal: `hp_local = amp_local[p] / sqrt(2)`
5. Search for crossings **within the local band only**, with linear interpolation
6. If no valid crossings: report Q=NaN, flag as "no clean resonance"
7. Fallback features for all bands: band energy, spectral centroid, local amplitude slope, phase-gradient magnitude, peak prominence, I/Q trajectory metrics
8. Add actual spectral shape features: centroid, variance, skewness, kurtosis of full-band amplitude envelope
9. Generate diagnostic plots: one figure per sweep showing all bands with peak marker, half-power level, crossing markers

I'll rerun NB5 (feature ML baselines) after the fix. RF may drop from 93.3% if some previously discriminative features were artifacts. **That's fine — physical correctness matters more than any headline number.**

---

## Point 4 — Validation split not sweep-independent

### Status: ACCEPTED — current val split has window-level leakage

**Confirmed from code inspection.** The current `build_dataset()` in `data.py` does:

```python
# split_indices("grouped"): holds out last sweep per class as test
train_idx, test_idx = split_indices(raw_labels, groups, split="grouped")

# Then randomly splits remaining windows 80/20 train/val
x_train, x_val, y_train, y_val, tg_train, tg_val = train_test_split(
    x_train_full, y_train_full, train_groups_full,
    test_size=0.2, random_state=42, stratify=y_train_full,
)
```

Since there are exactly 2 sweeps in `train_groups_full` (sweep 1 and sweep 2 after sweep 3 is held out), the random 80/20 split operates at the **window level** — windows from the same sweep appear in both train and val. The validation set is not independent of training acquisitions.

**Action:** Implement your proposed 3-fold sweep-level rotation:

| Fold | Train | Validation | Test |
|------|-------|------------|------|
| 1 | Sweep 1 | Sweep 2 | Sweep 3 |
| 2 | Sweep 2 | Sweep 3 | Sweep 1 |
| 3 | Sweep 3 | Sweep 1 | Sweep 2 |

- New function `build_dataset_folds()` returns `list[DatasetBundle]` — one per fold
- Each fold: normalization fitted on training windows only, applied to val + test
- Final result: mean ± std across 3 folds
- This prevents the result from depending on sweep 3 being chosen as the only test acquisition

---

## Point 5 — All models at the same prediction level

### Status: ACCEPTED — sweep-level aggregation needed

**Confirmed.** RF produces one feature vector per sweep (sweep-level by construction). CNN/SNN/ESN/LSM produce window-level predictions. The current comparison table mixes levels, making direct comparison invalid.

**Action:**
- New function `aggregate_to_sweep()` in `train_utils.py`:
  - Groups per-window predictions by `sweep_key` (from `test_groups`)
  - Default method: average class probabilities across windows → argmax
  - Alternative: majority vote
- Add `predict_proba()` to `LiquidStateMachine` (ESN already has it via `decision_function` → softmax)
- Main comparison table uses sweep-level accuracy; window-level is secondary
- Models without natural probability outputs (e.g., some SNN variants) will use majority vote

This is the correct approach. Window-level accuracy can still be reported as a diagnostic to show within-sweep consistency.

---

## Point 6 — Neural models need multiple random seeds

### Status: ACCEPTED — will run ≥5 seeds

**Current state:** `train.py` supports `--num-seeds` with `DEFAULT_SEEDS = [7, 42, 123, 999, 2024]` (5 seeds). NB11 generates seeds as `base_seed + i * 37`. Most saved results are single-seed.

**Action:**
- Default `--num-seeds 5` everywhere
- NB11: use the same explicit seed list as `train.py` (not `base_seed + i * 37`)
- Combine with 3-fold rotation: each fold × seed = 15 independent runs per model
- Report: accuracy mean ± std across all fold–seed combinations
- Distinguish fold variation from seed variation in reporting

---

## Point 7 — Explicit spike-encoded SNN performs poorly

### Status: ACKNOWLEDGED — will investigate root cause

**Current numbers:** Rate encoder→SNN = 32.5%, latency = 31.7%, delta = 28.3% (chance = 20%). This is ~10 pp above chance but ~3× below CNN1D (84%).

**What I'll investigate:**
1. Which encoder? Test all three (rate/latency/delta) with identical SNN config
2. Amplitude-only vs adding phase channel
3. Spike sparsity: what fraction of time steps have spikes?
4. Timing preservation: is frequency-sequential order maintained in tensor dimensions?
5. Binary events vs statistics: is the SNN receiving raw spike trains or aggregate features?
6. Information bottleneck: mutual information between encoded spikes and labels

**Possible outcomes:**
- If fixable (e.g., add phase, adjust encoding parameters): improve and re-report
- If fundamental: document as a **negative result** — "explicit spike conversion loses information for overdamped PMUT sensors where amplitude variation is < 5% across the sweep." This is a valid and useful scientific finding.

I agree this should not be presented as a competitive pipeline in its current state.

---

## Point 8 — ESN and LSM interpretation should be corrected

### Status: ACCEPTED — CLAUDE.md claims were wrong

**You're correct.** The saved results show ESN 80.9% ± 0.4% and LSM membrane 80.3% ± 1.4% on 5-class high-glucose. The LSM does NOT outperform ESN on the primary task. The previous CLAUDE.md claim ("LSM with membrane readout is the best neural model — beating CNN1D on both tasks") was based on the 13-class glucose-ladder where LSM did show +11 pp advantage — but that task is being removed from the main study (Point 1).

**The correct interpretation:** On 5-class high-glucose, ESN and LSM achieve broadly similar performance, with ESN being slightly more stable (lower variance). The LSM advantage on glucose-ladder was real but cannot be used as evidence of general superiority because the low-concentration portion lacks independent test sweeps.

**Action:** Update CLAUDE.md immediately. The Goal 3 report's narrative text is partially hardcoded — I'll replace it with dynamic text derived from actual metrics. The `_build_esn_lsm_row()` function already reads actual data correctly.

---

## Point 9 — Phase preprocessing needs verification

### Status: ACCEPTED — will audit systematically

**What I know from code inspection:**
- `high_glucose_xlsx`: reads `phase_deg` column → degrees ✓
- `low_glucose_xlsx`: reads `Phase (rad)` column, converts via `rad2deg` ✓
- X/Y for high: raw `x`/`y` columns (direct measurement) ✓
- X/Y for low: `amplitude * cos(phase_rad)` / `amplitude * sin(phase_rad)` (reconstructed from polar)

**What needs checking:**
1. Phase wrapping: ±180° discontinuities within 128-pt windows — if a wrap occurs inside a window, the CNN/LSM sees a step-function artifact
2. Sign convention: verify X/Y signs consistent across source files
3. Discontinuities per window: count and warn

**Action:**
- Add `validate_phase_data()` to `data.py` that checks wrapping within windows and prints warnings
- Add diagnostic plots: raw phase traces per class with wrap annotations
- The poor phase-only results (43.5% ESN, 38.7% LSM) may be genuinely weak information, but I agree we should eliminate preprocessing artifacts before concluding this

---

## Point 10 — Central results generator is incomplete

### Status: ACCEPTED — multiple bugs confirmed

**Specific bugs found during codebase exploration:**

1. **Hardcoded `_seed7`** (line 11): `glob('..._seed7/metrics.json')` only collects single-seed seed-7 results. Multi-seed runs with other seeds are invisible.

2. **LSM glob bug** (line 30): `{subdir}metrics_summary.json` with `subdir="lsm"` produces `lsmmetrics_summary.json` (missing `/`), so LSM results may not be found at all.

3. **Text-embedded numbers:** Accuracy stored as `"0.8087 ± 0.0035"` in one column. Reviewer wants separate `accuracy_mean`, `accuracy_std`, `num_seeds` columns.

4. **Missing model types:** No LSM spike readout, no fixed-point results (NB14), no encoder-comparison (NB9), no HDL verification status.

5. **`collect_reservoir_results()` in NB12** labels everything as `"ESN (n=...)"` — does not distinguish ESN from LSM.

6. **Modality key mismatch in NB12:** `plot_modality_summary()` expects `"amplitude_only"` but NB11 produces `"amplitude"`.

**Action:**
- Fix glob patterns (proper path join with `/`)
- Scan all seed directories
- Structured CSV with separate numeric columns: `accuracy_mean`, `accuracy_std`, `num_seeds`, `fold`, `prediction_level`
- Add sections for all model types (neural, reservoir ESN, reservoir LSM membrane, reservoir LSM spikes, encoder, fixed-point, feature ML)
- Each row includes: `result_file_path`, `git_commit`
- Fix modality key alignment between NB11 and NB12

---

## Point 11 — Fixed-point study is not end-to-end

### Status: ACCEPTED — will rename and build true integer model

**Current state confirmed:**
- Reservoir weights and membrane updates are quantized
- BUT: `FixedPointLIF._dot_product()` converts to float for the dot product (line 163: `fixed_to_float(a) @ fixed_to_float(b).T`)
- A new `RidgeClassifier` is trained in float at each bit width
- The readout weights, bias, accumulators, and argmax are NEVER quantized

**The current experiment answers:** "Can quantized reservoir features retain class information when followed by a retrained floating-point readout?" → YES, at 16-bit.

**It does NOT demonstrate a complete fixed-point classifier.**

**Action:**
1. **Rename:** "Retrained-Readout Reservoir Quantization Study" — update CLAUDE.md, NB14 docstring, GOAL_4_COMPLETION_REPORT.md
2. **Build `IntegerLSM` class** in `fixed_point.py`:
   - All operations in integer arithmetic — no float casts in forward pass
   - Explicit: multiplier width, accumulator width, rescaling (right-shift), rounding (truncation), overflow (saturation)
   - Quantized readout: Ridge weights → Q-format integers → integer dot product → argmax
   - Compare IntegerLSM vs float reference; measure accuracy delta
3. Also fix `FixedPointLIF` — it's currently a non-recurrent single-neuron primitive, not a full reservoir. The true integer model needs W_res recurrence in fixed-point.

The 16-bit result being +1–3% above float is within seed variance from retrained readouts — I'll stop attributing it to "implicit regularization."

---

## Point 12 — Verilog LIF neuron has update-order problem

### Status: ✅ CONFIRMED — serious bug found

**You're exactly right, and it's worse than it first appears.**

The current Verilog (lines 44-64 of `lif_neuron_fxp.v`) has MULTIPLE non-blocking assignments (`<=`) to `membrane` in the same `always` block:

```verilog
membrane <= (mult_temp >>> FRACTIONAL_BITS) + $signed(i_current);  // line 46 — NB
membrane <= MAX_VAL;    // line 50 — NB (dead code)
membrane <= MIN_VAL;    // line 52 — NB (dead code)
membrane <= $signed(membrane) - $signed(THRESHOLD);  // line 57 — NB (wins)
```

In Verilog, when multiple NBAs target the same register, **only the last assignment in source-order takes effect.** This means:
- The leak+integrate result (line 46) is DISCARDED
- The saturation code (lines 50, 52) is DEAD CODE
- The spike+reset (line 57) operates on the OLD `membrane` value — the one from the previous clock cycle
- Effectively: if spike, `membrane = old_membrane - THRESHOLD`; if no spike, `membrane` is unchanged

**Fix using `v_next` combinational logic:**
```verilog
reg [TOTAL_BITS-1:0] v_next;
// Compute leak+integrate
mult_temp = $signed(BETA) * $signed(membrane);
v_next = (mult_temp >>> FRACTIONAL_BITS) + $signed(i_current);
// Saturate
if ($signed(v_next) > $signed(MAX_VAL))
    v_next = MAX_VAL;
else if ($signed(v_next) < $signed(MIN_VAL))
    v_next = MIN_VAL;
// Spike + reset
if ($signed(v_next) >= $signed(THRESHOLD)) begin
    o_spike <= 1'b1;
    v_next = $signed(v_next) - $signed(THRESHOLD);
end else
    o_spike <= 1'b0;
// SINGLE non-blocking assignment
membrane <= v_next;
o_membrane <= v_next;
```

This ensures: (a) all three stages compose correctly, (b) the spike check uses the newly computed value, (c) only one NBA to `membrane`. I'll also regenerate this from NB16 since the file header says "Auto-generated by NB16."

---

## Point 13 — HDL testbench doesn't perform verification

### Status: ACCEPTED — skeleton only

**Confirmed.** The testbench has:
- Clock generation ✓
- Reset sequence ✓
- `$dumpfile`/`$dumpvars` for waveform viewing ✓
- **Comment placeholders** for `$readmemb` — no data loaded ✗
- **No assertions or comparisons** ✗
- **No pass/fail** ✗
- `#1000 $finish` — runs for 1000 time units and exits regardless of correctness ✗

**Action:**
1. **Rewrite testbench** with actual self-checking:
   - `$readmemh` to load PMUT input spikes, expected voltage, expected spikes
   - Cycle-by-cycle comparison with tolerance (±1 LSB for voltage, exact for spikes)
   - Count mismatches
   - Report `PASS: N/M cycles match` or `FAIL: N mismatches at cycles [c1, c2, ...]`
2. **New `verify_hdl_simulation.py`**: try to run `iverilog` + `vvp` if installed; compare Python FixedPointLIF output against Verilog output
3. Update `verify_hdl_checklist.py` to include simulation-based checks

---

## Point 14 — PMUT-spike-to-current scaling

### Status: ✅ CONFIRMED — scaling bug found

**The bug** (in `generate_hdl_test_vectors()`, `fixed_point.py` lines 468-476):

```python
s_in_q = input_spikes[t].astype(np.int64)   # raw binary 0 or 1
dot = np.sum(W_in_q[j].astype(np.int64) * s_in_q)  # Q-format weight × binary 0/1
I_q[j] = bias_q[j] + (dot // (2 ** qfmt.fractional_bits))  # DIVIDES by 2^12
```

`W_in_q[j,i]` is already in Q-format (e.g., Q3.12 → integer in range ~[-32768, 32767]). Multiplying by binary 0/1 gives either 0 or the full Q-format integer. Then dividing by `2^12 = 4096` scales it DOWN. A spike of 1 should deliver the FULL Q-format weight, not weight ÷ 4096.

This means: the input spike contribution is attenuated by a factor of 4096 below the bias contribution. The neuron's response is dominated by bias, not by the PMUT input. Different input spike patterns can produce identical output because the input contribution is below the LSB.

**Fix:** Binary spike = gate — no rescaling needed:
```python
I_q[j] = bias_q[j]
for i in range(n_input):
    if s_in_q[i]:
        I_q[j] += W_in_q[j, i]  # Direct addition — weight already in Q-format
```

After the fix, different PMUT spike input files MUST produce different output spike sequences. This also affects the Verilog weight ROM design — the hardware must not apply an extra Q-format rescaling to the spike-gated weights.

---

## Point 15 — Recommended order of work

### Status: ACKNOWLEDGED — will follow this order

Your recommended priority is exactly right: software + experiment fixes first, then hardware.

**My implementation order:**

| Phase | Points | What |
|-------|--------|------|
| **1** | 1, 4, 5 | Remove low-conc, 3-fold sweep rotation, sweep-level aggregation |
| **2** | 2 | Freeze clean commit, move outputs → outputs_legacy, REPRODUCIBILITY.md |
| **3** | 3 | Fix PMUT feature extraction (bandwidth/Q), add shape features |
| **4** | 6, 8 | Multi-seed (5), ESN/LSM interpretation correction |
| **5** | 10 | Fix central results CSV (globs, structured columns, all model types) |
| **6** | 9 | Phase preprocessing audit + validation |
| **7** | 7 | Diagnose encoder→SNN pipeline |
| **8** | 11, 14 | IntegerLSM, fix spike-to-current scaling |
| **9** | 12, 13 | Verilog LIF fix, self-checking testbench |
| **10** | — | Regenerate ALL results, update CLAUDE.md |

I will not touch HDL until the software experiments and feature extraction are frozen. I completely agree that fixing the Verilog before the model architecture is stable would be wasted effort.

---

## Additional bugs found during codebase exploration

These were found while investigating your review points. They are not in your 15-point list but affect correctness:

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| A | `collect_reservoir_results()` labels LSM as ESN | `nb12_model_comparison.py:153-189` | LSM results mislabeled in comparison table |
| B | Modality key mismatch | NB11 produces `"amplitude"`, NB12 expects `"amplitude_only"` | Modality plot silently produces empty bars |
| C | `build_central_results.py` only scans `_seed7` | Line 11 | Multi-seed results invisible to central CSV |
| D | `FixedPointLIF._dot_product()` converts to float | `fixed_point.py:163` | Not true fixed-point — uses float intermediate |
| E | `FixedPointLIF` has no recurrence (W_res) | `fixed_point.py:106` | Cannot model full reservoir — single-neuron only |

All of these are included in the implementation plan.

---

## Rebuttals / Discussion Points

**On Point 1 (removing low-concentration):** Fully agreed, no rebuttal. The single-sweep-per-class limitation is fundamental — no amount of window trickery can produce independent test data.

**On Point 3 (Q extraction):** You correctly identified that the "heavily overdamped" conclusion was based on artifact Q values. I'll remove that claim from CLAUDE.md immediately. Whether the PMUT is actually overdamped in liquid is a separate physical question that needs proper measurement, not an artifact-prone extraction.

**On Point 8 (ESN vs LSM):** Mea culpa — the CLAUDE.md claim was wrong. I should have checked the actual saved metrics rather than trusting earlier hardcoded numbers. The dynamic `_build_esn_lsm_row()` function was reading correct data all along — I just didn't propagate it to the narrative text.

**On Point 12 (Verilog):** This is the most concerning bug. Thank you for catching it. The multiple-NBA issue means the Verilog would synthesize to completely wrong behavior. I'll fix this with the `v_next` pattern and verify against the Python FixedPointLIF cycle-by-cycle.

**On the overall direction:** I agree the 5-class high-glucose study is the strongest structure. 50–1000 mM with 3 sweeps each, strict sweep-level evaluation, honest train/val/test separation — this is a defensible result. Everything else (low-concentration, encoder→SNN, full fixed-point readout) can be follow-up work.

---

## Summary

| Your Point | My Response | Action |
|------------|-------------|--------|
| 1. Remove low-conc | **Agreed** | Remove from main study; keep for visualization |
| 2. Mixed code versions | **Agreed** | Clean freeze + regenerate all |
| 3. Q/bw extraction | **Agreed + bug confirmed** | Rewrite with local-band method |
| 4. Val not sweep-independent | **Agreed** | 3-fold sweep rotation |
| 5. Different prediction levels | **Agreed** | Sweep-level aggregation for all models |
| 6. Multi-seed needed | **Agreed** | Default 5 seeds × 3 folds |
| 7. Encoder→SNN poor | **Agreed** | Diagnose; document as negative result if fundamental |
| 8. ESN/LSM interpretation | **Agreed + my error** | Correct CLAUDE.md; fix hardcoded narrative |
| 9. Phase preprocessing | **Agreed** | Audit + validate + diagnostic plots |
| 10. Central results broken | **Agreed + bugs found** | Fix globs, structured CSV, all models |
| 11. Fixed-point not e2e | **Agreed** | Rename study; build IntegerLSM |
| 12. Verilog update-order | **Agreed + confirmed serious** | Fix with v_next; verify cycle-by-cycle |
| 13. Testbench doesn't verify | **Agreed** | Self-checking testbench + simulation |
| 14. Spike-to-current scaling | **Agreed + bug confirmed** | Fix — binary spike = gate, no rescaling |
| 15. Order of work | **Agreed** | Software first, HDL last |

As always, thank you for the thorough review. I'd rather fix all these issues now than publish results we'd have to retract later. I'll send the branch/commit hash once the clean freeze is done.

If any of my responses seem wrong, or if I've missed nuance in any of your points, please push back — we're co-collaborators and I should be catching your mistakes too, not just following instructions.
