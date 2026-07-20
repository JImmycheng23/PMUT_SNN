"""
Notebook 4: PMUT Feature Extraction
=====================================
Extract physically interpretable features from each PMUT frequency sweep:
  - Resonance frequency (f₀)
  - Peak amplitude
  - Phase at resonance
  - Phase transition frequency & slope
  - Bandwidth (Δf at -3 dB points)
  - Q-factor = f₀ / Δf
  - Frequency shift relative to Air baseline
  - Amplitude change relative to Air baseline
  - Area under amplitude curve

Run:  python notebooks/nb4_feature_extraction.py [--task pure]
Saves: outputs/features/{task}_feature_table.csv
       outputs/features/{task}_feature_distributions.png
       outputs/features/{task}_pca_umap.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- matplotlib setup -------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import (
    load_task_frame,
    SIGNAL_COLUMNS,
    _label_sort_key,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "features"


# ---------------------------------------------------------------------------
# 1. Per-sweep feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_sweep(
    freq: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    label: str,
    sweep_key: str,
) -> dict:
    """Extract all physical features from a single sweep."""
    n = len(freq)
    if n < 4:
        return {}

    # ---- resonance ---------------------------------------------------------
    peak_idx = int(np.argmax(amp))
    f0 = float(freq[peak_idx])
    amp_peak = float(amp[peak_idx])
    phase_at_res = float(phase[peak_idx])

    # ---- bandwidth (-3 dB = amp / sqrt(2)) --------------------------------
    half_power = amp_peak / np.sqrt(2)
    # left side: find crossing by searching from peak outward
    left_idx = 0
    for i in range(peak_idx, 0, -1):
        if amp[i] <= half_power:
            left_idx = i
            break
    # right side
    right_idx = n - 1
    for i in range(peak_idx, n):
        if amp[i] <= half_power:
            right_idx = i
            break
    # If crossing points found, interpolate for precision
    f_left = float(freq[left_idx])
    f_right = float(freq[right_idx])
    if left_idx > 0 and amp[left_idx] < half_power and amp[left_idx + 1] > half_power:
        # interpolate between left_idx and left_idx+1
        a1, a2 = amp[left_idx], amp[left_idx + 1]
        if abs(a2 - a1) > 1e-20:
            frac = (half_power - a1) / (a2 - a1)
            f_left = float(freq[left_idx] + frac * (freq[left_idx + 1] - freq[left_idx]))
    if right_idx < n - 1 and amp[right_idx] < half_power and amp[right_idx - 1] > half_power:
        a1, a2 = amp[right_idx - 1], amp[right_idx]
        if abs(a2 - a1) > 1e-20:
            frac = (half_power - a1) / (a2 - a1)
            f_right = float(freq[right_idx - 1] + frac * (freq[right_idx] - freq[right_idx - 1]))
    delta_f = max(f_right - f_left, 1.0)  # avoid div by zero
    q_factor = f0 / delta_f

    # ---- phase transition --------------------------------------------------
    # find the frequency range where phase changes most rapidly
    phase_diff = np.abs(np.diff(phase))
    # unwrap large jumps > 300 deg as likely wrapping artifacts
    transition_window = 20  # points around resonance
    start_t = max(0, peak_idx - transition_window)
    end_t = min(n, peak_idx + transition_window)
    phase_slope = float(np.max(np.abs(np.diff(phase[start_t:end_t]))))

    # phase transition frequency: where phase crosses -90° (or midpoint)
    phase_mid = (phase.max() + phase.min()) / 2
    mid_idx = int(np.argmin(np.abs(phase - phase_mid)))
    f_phase_transition = float(freq[mid_idx])

    # ---- amplitude statistics ----------------------------------------------
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_auc = float(np.trapezoid(amp, freq))  # area under curve

    # ---- X/Y channel statistics -------------------------------------------
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))

    # ---- noise floor (median of lowest 20% amplitudes) --------------------
    low_cut = int(n * 0.2)
    noise_floor = float(np.median(np.sort(amp)[: max(low_cut, 1)]))

    return {
        "sweep_key": sweep_key,
        "label": label,
        "n_points": n,
        "f0_hz": f0,
        "amp_peak": amp_peak,
        "phase_at_resonance_deg": phase_at_res,
        "bandwidth_hz": delta_f,
        "q_factor": q_factor,
        "phase_slope_deg_per_bin": phase_slope,
        "f_phase_transition_hz": f_phase_transition,
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "amp_auc": amp_auc,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "noise_floor": noise_floor,
        "freq_min_hz": float(freq.min()),
        "freq_max_hz": float(freq.max()),
    }


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
    # sort by label
    df["_sort"] = df["label"].apply(
        lambda lb: _label_sort_key(lb)[1] if isinstance(_label_sort_key(lb), tuple) and _label_sort_key(lb)[0] == 1 else 0
    )
    df = df.sort_values(["label", "sweep_key"]).drop(columns=["_sort"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Baseline-relative features
# ---------------------------------------------------------------------------

def add_baseline_features(df: pd.DataFrame, baseline_label: str = "Air") -> pd.DataFrame:
    """Add features measured relative to a baseline (default: Air)."""
    baseline = df[df["label"] == baseline_label]
    if baseline.empty:
        print(f"Warning: no '{baseline_label}' sweeps found; skipping baseline features.")
        return df

    # use median of baseline sweeps for robustness
    f0_ref = baseline["f0_hz"].median()
    amp_ref = baseline["amp_peak"].median()
    q_ref = baseline["q_factor"].median()

    df["f0_shift_hz"] = df["f0_hz"] - f0_ref
    df["f0_shift_pct"] = (df["f0_hz"] - f0_ref) / f0_ref * 100
    df["amp_change_ratio"] = df["amp_peak"] / amp_ref
    df["q_change_ratio"] = df["q_factor"] / q_ref

    return df


# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Box-plots and pairwise scatter of key features."""
    features = [
        ("f0_hz", "Resonance Frequency (Hz)"),
        ("amp_peak", "Peak Amplitude (V)"),
        ("bandwidth_hz", "Bandwidth (Hz)"),
        ("q_factor", "Q-Factor"),
        ("phase_slope_deg_per_bin", "Phase Slope (°/bin)"),
        ("amp_auc", "Amplitude AUC"),
        ("noise_floor", "Noise Floor (V)"),
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
    print(f"Saved feature distributions → {out_dir / f'{task}_feature_distributions.png'}")


def plot_pca_umap(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """PCA and UMAP dimensionality reduction of the feature space."""
    feature_cols = [
        "f0_hz", "amp_peak", "bandwidth_hz", "q_factor",
        "phase_slope_deg_per_bin", "amp_auc", "noise_floor",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    # fill any NaN with median
    X = X.fillna(X.median())

    # standardize
    X_std = (X - X.mean()) / X.std(ddof=0)
    labels_list = df["label"].tolist()
    unique_labels = sorted(set(labels_list), key=_label_sort_key)
    cmap = {lb: plt.cm.tab10(i) for i, lb in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- PCA ---
    from numpy.linalg import svd
    U, S, Vt = svd(X_std.values - X_std.values.mean(axis=0), full_matrices=False)
    pca = (U[:, :2] * S[:2])
    ax = axes[0]
    for lb in unique_labels:
        mask = [l == lb for l in labels_list]
        ax.scatter(pca[mask, 0], pca[mask, 1], c=[cmap[lb]], label=lb, s=60, alpha=0.8, edgecolors="k", linewidth=0.5)
    var_explained = (S[:2] ** 2) / (S**2).sum() * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_title("PCA — Feature Space")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- UMAP (optional, skip if not installed) ---
    ax2 = axes[1]
    try:
        import umap
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(X_std.values)
        for lb in unique_labels:
            mask = [l == lb for l in labels_list]
            ax2.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap[lb]], label=lb, s=60, alpha=0.8, edgecolors="k", linewidth=0.5)
        ax2.set_title("UMAP — Feature Space")
        ax2.legend(fontsize=8)
    except ImportError:
        ax2.text(0.5, 0.5, "UMAP not installed\n(pip install umap-learn)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_title("UMAP (unavailable)")

    fig.suptitle(f"PMUT Feature Space Dimensionality Reduction — {task}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_pca_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PCA/UMAP → {out_dir / f'{task}_pca_umap.png'}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print class-wise feature statistics."""
    print("\n" + "=" * 80)
    print("CLASS-WISE FEATURE SUMMARY (mean ± std)")
    print("=" * 80)
    feats = ["f0_hz", "amp_peak", "bandwidth_hz", "q_factor", "phase_slope_deg_per_bin"]
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
    plot_pca_umap(df, args.task, out_dir)

    print("\nDone — Notebook 4 complete.")


if __name__ == "__main__":
    main()
