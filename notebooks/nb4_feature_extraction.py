"""
Notebook 4: PMUT Feature Extraction
=====================================
Extract physically interpretable features from each PMUT frequency sweep.

Tasks:
  - Detect resonance peaks (4-band hybrid multi-modal detection)
  - Estimate resonance frequency (f0) for each mode
  - Estimate peak amplitude at each resonance
  - Estimate phase transition location and slope
  - Calculate bandwidth (-3dB) and Q-factor for each mode
  - Calculate frequency shift relative to Air/DI Water baseline

Run:  python notebooks/nb4_feature_extraction.py [--task high-glucose]
Saves: outputs/features/{task}_feature_table.csv
       outputs/features/{task}_feature_distributions.png
       outputs/features/{task}_feature_correlation.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import load_task_frame, _label_sort_key

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "features"


# ---------------------------------------------------------------------------
# Helper: banded hybrid multi-peak detection
# ---------------------------------------------------------------------------

# Four expected PMUT vibration mode bands (from domain knowledge)
PMUT_MODE_BANDS = [
    {"name": "b1", "label": "0.2M", "range_hz": (120_000, 380_000),  "method": "phase"},
    {"name": "b2", "label": "0.45M", "range_hz": (380_000, 700_000),  "method": "amp"},
    {"name": "b3", "label": "0.9M",  "range_hz": (700_000, 1_100_000), "method": "amp"},
    {"name": "b4", "label": "1.2M",  "range_hz": (1_100_000, 2_450_000), "method": "amp"},
]


def _detect_banded_peaks(
    freq: np.ndarray, amp: np.ndarray, phase: np.ndarray, xv: np.ndarray, yv: np.ndarray,
) -> dict:
    """Detect up to 4 PMUT resonance modes using band-specific hybrid methods.

    Returns per-band: f0_bN_hz, q_bN, amp_bN, bw_bN_hz, peak_bN.
    """
    n = len(freq)
    freq_mhz = freq / 1e6
    result = {}

    # Stage A: Amplitude-based peak finding (deg-2 polynomial detrend)
    coeffs = np.polyfit(freq_mhz, amp, 2)
    bg = np.polyval(coeffs, freq_mhz)
    amp_det = amp - bg
    det_range = np.ptp(amp_det)

    min_prom = max(det_range * 0.06, 1e-12)
    min_dist = max(int(n * 0.015), 5)
    peaks_amp, props_amp = find_peaks(amp_det, prominence=min_prom, distance=min_dist, width=3)
    amp_proms = props_amp.get("prominences", [])

    # Stage B: Phase-derivative peak finding (for deep 0.2 MHz mode)
    win = max(n // 12 * 2 + 1, 51)
    if win % 2 == 0:
        win += 1
    try:
        phase_sm = savgol_filter(phase, window_length=min(win, n - 1 - (n % 2)), polyorder=3)
    except Exception:
        phase_sm = phase
    dphase = np.abs(np.gradient(phase_sm, freq))
    dphase[: int(n * 0.03)] = 0
    dphase[int(n * 0.97):] = 0

    peaks_ph, props_ph = find_peaks(dphase, prominence=np.std(dphase) * 0.5, distance=min_dist)
    ph_proms = props_ph.get("prominences", [])

    for band in PMUT_MODE_BANDS:
        bname = band["name"]
        lo, hi = band["range_hz"]
        method = band["method"]

        # Gather amplitude candidates in this band
        amp_cands = []
        for j, p in enumerate(peaks_amp):
            f = float(freq[p])
            if lo <= f <= hi:
                hp = amp[p] / np.sqrt(2)
                left = 0
                for k in range(p, 0, -1):
                    if amp[k] <= hp:
                        left = k
                        break
                right = n - 1
                for k in range(p, n):
                    if amp[k] <= hp:
                        right = k
                        break
                bw = float(freq[right] - freq[left])
                q = f / bw if bw > 0 else 0.0
                amp_cands.append({
                    "f": f, "q": q, "bw": bw,
                    "prom": float(amp_proms[j]) if len(amp_proms) > j else 0,
                    "amp_at_peak": float(amp[p]),
                })

        # Gather phase-derivative candidates in this band
        phase_cands = []
        for j, p in enumerate(peaks_ph):
            f = float(freq[p])
            if lo <= f <= hi:
                phase_cands.append({
                    "f": f, "q": 0.0, "bw": 0.0,
                    "prom": float(ph_proms[j]) if len(ph_proms) > j else 0,
                    "amp_at_peak": float(amp[p]),
                })

        best = None
        if method == "amp" and amp_cands:
            best = max(amp_cands, key=lambda c: c["prom"])
        elif method == "phase":
            if phase_cands:
                best = max(phase_cands, key=lambda c: c["prom"])
            elif amp_cands:
                best = max(amp_cands, key=lambda c: c["prom"])
        elif amp_cands:
            best = max(amp_cands, key=lambda c: c["prom"])

        if best is not None:
            result[f"f0_{bname}_hz"] = best["f"]
            result[f"q_{bname}"] = best["q"]
            result[f"amp_{bname}"] = best["amp_at_peak"]
            result[f"bw_{bname}_hz"] = best["bw"]
            result[f"peak_{bname}"] = True
        else:
            result[f"f0_{bname}_hz"] = np.nan
            result[f"q_{bname}"] = np.nan
            result[f"amp_{bname}"] = np.nan
            result[f"bw_{bname}_hz"] = np.nan
            result[f"peak_{bname}"] = False

    return result


# ---------------------------------------------------------------------------
# Helper: phase transition features
# ---------------------------------------------------------------------------

def _extract_phase_features(freq: np.ndarray, phase: np.ndarray) -> dict:
    """Extract phase transition location and slope.

    The phase changes most rapidly near resonance. We find the frequency
    of steepest negative phase slope (smoothed) and measure the slope there.
    """
    n = len(freq)
    win = max(n // 12 * 2 + 1, 51)
    if win % 2 == 0:
        win += 1
    try:
        phase_sm = savgol_filter(phase, window_length=min(win, n - 1 - (n % 2)), polyorder=3)
    except Exception:
        phase_sm = phase

    dphase_df = np.gradient(phase_sm, freq)
    # Exclude boundaries
    lo, hi = int(n * 0.05), int(n * 0.95)
    search = dphase_df.copy()
    search[:lo] = 0
    search[hi:] = 0

    # Steepest negative slope = most negative dphase/df value
    idx = int(np.argmin(search))
    f_transition = float(freq[idx])
    slope = float(dphase_df[idx])

    return {
        "f_phase_transition_hz": f_transition,
        "phase_slope_deg_per_hz": abs(slope),
    }


# ---------------------------------------------------------------------------
# 1. Per-sweep feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_sweep(
    freq: np.ndarray, amp: np.ndarray, phase: np.ndarray,
    xv: np.ndarray, yv: np.ndarray,
    label: str, sweep_key: str,
) -> dict:
    """Extract resonance, phase, and level features from a single PMUT sweep."""
    n = len(freq)
    if n < 4:
        return {}

    # ---- 1. Banded multi-peak resonance detection --------------------------
    banded = _detect_banded_peaks(freq, amp, phase, xv, yv)

    # ---- 2. Phase transition features --------------------------------------
    phase_feat = _extract_phase_features(freq, phase)

    # ---- 3. Overall level features ----------------------------------------
    amp_mean = float(np.mean(amp))
    amp_auc = float(np.trapezoid(amp, freq))
    peak_idx = int(np.argmax(amp))
    amp_peak = float(amp[peak_idx])

    return {
        "sweep_key": sweep_key,
        "label": label,
        "n_points": n,
        # Resonance peak features (4 PMUT modes)
        **banded,
        # Phase transition features
        "f_phase_transition_hz": phase_feat["f_phase_transition_hz"],
        "phase_slope_deg_per_hz": phase_feat["phase_slope_deg_per_hz"],
        # Overall level
        "amp_mean": amp_mean,
        "amp_peak": amp_peak,
        "amp_auc": amp_auc,
    }


# ---------------------------------------------------------------------------
# 2. Baseline-relative features
# ---------------------------------------------------------------------------

def add_baseline_features(df: pd.DataFrame, baseline_label: str = "Air") -> pd.DataFrame:
    """Add frequency shift features relative to a baseline (Air or DI Water).

    Computes f0_shift_hz and f0_shift_pct for each band's resonance frequency
    relative to the baseline median.
    """
    baseline = df[df["label"] == baseline_label]
    if baseline.empty:
        print(f"  Note: no '{baseline_label}' sweeps found; skipping baseline features.")
        return df

    band_names = ["b1", "b2", "b3", "b4"]
    for bname in band_names:
        col = f"f0_{bname}_hz"
        if col not in baseline.columns:
            continue
        ref = baseline[col].median()
        if np.isfinite(ref) and ref > 0:
            for idx in df.index:
                val = df.loc[idx, col]
                if np.isfinite(val):
                    df.loc[idx, f"f0_{bname}_shift_hz"] = val - ref
                    df.loc[idx, f"f0_{bname}_shift_pct"] = (val - ref) / ref * 100

    return df


# ---------------------------------------------------------------------------
# 3. Build feature table
# ---------------------------------------------------------------------------

def build_feature_table(task: str, data_root: Path) -> pd.DataFrame:
    """Load the task frame and extract features for every sweep."""
    frame = load_task_frame(task, data_root)
    rows = []
    for (_, sweep_key, label), group in frame.groupby(
        ["source", "sweep_key", "label"], sort=False
    ):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        freq = group["frequency_hz"].to_numpy(dtype=float)
        amp = group["amplitude"].to_numpy(dtype=float)
        phase = group["phase"].to_numpy(dtype=float)
        xv = group["x_v"].to_numpy(dtype=float)
        yv = group["y_v"].to_numpy(dtype=float)
        row = extract_features_from_sweep(freq, amp, phase, xv, yv, label, sweep_key)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    df["_sort"] = df["label"].apply(
        lambda lb: _label_sort_key(lb)[1]
        if isinstance(_label_sort_key(lb), tuple) and _label_sort_key(lb)[0] == 1
        else 0
    )
    df = df.sort_values(["label", "sweep_key"]).drop(columns=["_sort"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Box-plots of resonance and phase features per class."""
    features = [
        ("f0_b2_hz", "Resonance Freq — Mode 2 (0.45M, Hz)"),
        ("q_b2", "Q-Factor — Mode 2"),
        ("bw_b2_hz", "Bandwidth — Mode 2 (Hz)"),
        ("f0_b3_hz", "Resonance Freq — Mode 3 (0.9M, Hz)"),
        ("q_b3", "Q-Factor — Mode 3"),
        ("amp_peak", "Peak Amplitude"),
        ("amp_auc", "Amplitude AUC"),
        ("f_phase_transition_hz", "Phase Transition Frequency (Hz)"),
        ("phase_slope_deg_per_hz", "Phase Slope (deg/Hz)"),
    ]
    labels = sorted(df["label"].unique(), key=_label_sort_key)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, features):
        if col not in df.columns:
            continue
        data = [df[df["label"] == lb][col].dropna().values for lb in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PMUT Feature Distributions — {task}", fontsize=14, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature distributions → {out_dir / f'{task}_feature_distributions.png'}")


def plot_feature_correlation(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Feature correlation matrix."""
    feat_cols = [
        "f0_b1_hz", "q_b1", "f0_b2_hz", "q_b2",
        "f0_b3_hz", "q_b3", "f0_b4_hz", "q_b4",
        "amp_mean", "amp_peak", "amp_auc",
        "f_phase_transition_hz", "phase_slope_deg_per_hz",
    ]
    available = [c for c in feat_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(f"Feature Correlation Matrix — {task}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature correlation → {out_dir / f'{task}_feature_correlation.png'}")


def plot_class_wise_comparison(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Class-wise comparison of key resonance features (grouped bar plot)."""
    labels = sorted(df["label"].unique(), key=_label_sort_key)
    feat_cols = [("f0_b2_hz", "Mode 2 f0 (MHz)"), ("q_b2", "Mode 2 Q"), ("amp_peak", "Peak Amp")]

    fig, axes = plt.subplots(1, len(feat_cols), figsize=(5 * len(feat_cols), 4))
    if len(feat_cols) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, feat_cols):
        if col not in df.columns:
            continue
        means = [df[df["label"] == lb][col].mean() for lb in labels]
        stds = [df[df["label"] == lb][col].std() for lb in labels]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Class-Wise Feature Comparison — {task}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_class_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved class-wise comparison → {out_dir / f'{task}_class_comparison.png'}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print class-wise feature statistics."""
    print("\n" + "=" * 80)
    print("CLASS-WISE FEATURE SUMMARY (mean ± std)")
    print("=" * 80)
    feats = ["f0_b1_hz", "f0_b2_hz", "f0_b3_hz", "f0_b4_hz",
             "q_b2", "q_b3", "amp_mean", "amp_peak",
             "f_phase_transition_hz"]
    for lb in sorted(df["label"].unique(), key=_label_sort_key):
        sub = df[df["label"] == lb]
        parts = [f"{lb:<20s}"]
        for f in feats:
            if f in df.columns:
                parts.append(f"{f}={sub[f].mean():.4g} ± {sub[f].std():.4g}")
        print("  ".join(parts))
    print("=" * 80)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PMUT physical features.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features for task: {args.task}")
    df = build_feature_table(args.task, args.data_root)
    df = add_baseline_features(df, baseline_label="Air")

    csv_path = out_dir / f"{args.task}_feature_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved feature table ({len(df)} sweeps, {len(df.columns)} features) → {csv_path}")

    print_feature_summary(df)

    plot_feature_distributions(df, args.task, out_dir)
    plot_feature_correlation(df, args.task, out_dir)
    plot_class_wise_comparison(df, args.task, out_dir)

    print("\nDone — Notebook 4 complete.")


if __name__ == "__main__":
    main()
