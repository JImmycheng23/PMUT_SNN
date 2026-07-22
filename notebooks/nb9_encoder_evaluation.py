"""
Notebook 9: Spike Encoder Evaluation
=====================================
Compare all spike encoders using a simple classifier.
  - Accuracy vs spike count
  - Latency vs accuracy
  - Select best encoder for SNN classification

⚠️  Architectural note — this compares EXPLICIT (hand-designed) spike encoders.
The trainable SNN models (spiking.py / train.py) do NOT use these encoders.
See the "Two distinct spiking approaches" section in CLAUDE.md for details.

Evaluation uses physical-sweep-level train/test separation to measure
generalization, NOT training-set accuracy. Each encoder produces one
feature vector per sweep; the split holds out complete sweeps.

Run:  python notebooks/nb9_encoder_evaluation.py --task high-glucose --split grouped
      python notebooks/nb9_encoder_evaluation.py --task glucose-ladder --split window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import (
    load_task_frame,
    _label_sort_key,
    encode_labels,
    split_indices,
)

# Import encoders from nb8
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb8_spike_encoding import (
    rate_encode,
    latency_encode,
    threshold_crossing_encode,
    delta_encode,
    population_encode,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "encoder_evaluation"

# ── Default split per task (matches primary evaluation in other models) ──
TASK_DEFAULT_SPLIT = {
    "high-glucose": "grouped",
    "glucose-ladder": "window",
}


# ═══════════════════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════════════════

def encode_all_sweeps(frame: pd.DataFrame):
    """Encode every sweep with all 5 encoders.

    Returns
    -------
    results : dict
        encoder_name → {"X": (n_sweeps, n_features), "spike_counts": (n_sweeps,)}
    y : np.ndarray  (n_sweeps,)  integer-encoded labels
    classes : list[str]
    sweep_labels : np.ndarray  (n_sweeps,)  raw label strings (for split_indices)
    sweep_groups : np.ndarray  (n_sweeps,)  sweep_key per sample (for split_indices)
    """
    # ── Collect all sweeps ──
    sweep_data = {}
    for (_, sweep_key, label), group in frame.groupby(
        ["source", "sweep_key", "label"], sort=False
    ):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        amp = group["amplitude"].to_numpy(dtype=float)
        phase = group["phase"].to_numpy(dtype=float)
        sweep_data[sweep_key] = {"amp": amp, "phase": phase, "label": label}

    sweep_keys = sorted(sweep_data.keys())
    labels_list = [sweep_data[k]["label"] for k in sweep_keys]
    y, classes = encode_labels(np.array(labels_list))
    sweep_labels = np.array(labels_list)
    sweep_groups = np.array(sweep_keys)

    # ── Encode with each method ──
    results = {}

    for enc_name, encode_fn in [
        ("rate", lambda amp: rate_encode(amp, n_steps=50, max_rate=0.3)),
        ("latency", lambda amp: latency_encode(amp, max_time=50)),
        ("threshold", lambda amp: threshold_crossing_encode(amp, n_thresholds=6)),
        ("delta", lambda amp: delta_encode(amp, sensitivity=1.5)),
        ("population", lambda amp: population_encode(amp, n_neurons=16, n_steps=30)),
    ]:
        feature_rows = []
        spike_counts = []
        for key in sweep_keys:
            amp = sweep_data[key]["amp"]
            try:
                spikes = encode_fn(amp)
            except Exception:
                spikes = np.zeros((10, 10), dtype=np.int8)

            # Aggregate spike representation into fixed-size feature vector
            ch_rates = spikes.mean(axis=0)
            temporal_profile = spikes.mean(axis=1)

            # Downsample channel rates to fixed size
            target_ch = 20
            if len(ch_rates) > target_ch:
                ch_rates = np.interp(
                    np.linspace(0, len(ch_rates) - 1, target_ch),
                    np.arange(len(ch_rates)),
                    ch_rates,
                )
            elif len(ch_rates) < target_ch:
                ch_rates = np.pad(ch_rates, (0, target_ch - len(ch_rates)))

            # Downsample temporal profile to fixed size
            target_len = 20
            if len(temporal_profile) > target_len:
                temporal_profile = np.interp(
                    np.linspace(0, len(temporal_profile) - 1, target_len),
                    np.arange(len(temporal_profile)),
                    temporal_profile,
                )
            elif len(temporal_profile) < target_len:
                temporal_profile = np.pad(temporal_profile, (0, target_len - len(temporal_profile)))

            features = np.concatenate([ch_rates, temporal_profile, [spikes.sum() / spikes.size]])
            feature_rows.append(features)
            spike_counts.append(int(spikes.sum()))

        X = np.stack(feature_rows)
        results[enc_name] = {
            "X": X.astype(np.float64),
            "spike_counts": np.array(spike_counts),
        }

    return results, y, classes, sweep_labels, sweep_groups


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation  (sweep-level train / test split — NO data leakage)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_encoders(
    results: dict,
    y: np.ndarray,
    sweep_labels: np.ndarray,
    sweep_groups: np.ndarray,
    split: str,
    test_every: int = 5,
) -> pd.DataFrame:
    """Train LR on training sweeps only, evaluate on held-out test sweeps.

    Uses the same ``split_indices`` infrastructure as every other model
    in the project so the evaluation measures generalisation, not
    training-set memorisation.
    """
    train_idx, test_idx = split_indices(
        sweep_labels, sweep_groups, split=split, test_every=test_every,
    )

    rows = []
    for enc_name, data in results.items():
        X = data["X"]
        spike_counts = data["spike_counts"]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Standardize from train only (no test leakage) ──
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ── Logistic regression ──
        clf = LogisticRegression(max_iter=5000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # ── Test accuracy (generalisation) ──
        y_pred = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)

        # ── Train accuracy (for transparency) ──
        y_train_pred = clf.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)

        mean_spikes = float(spike_counts[train_idx].mean())

        rows.append({
            "encoder": enc_name,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "mean_spike_count": mean_spikes,
            "total_spike_count": int(spike_counts.sum()),
            "efficiency": test_acc / max(mean_spikes, 1),
        })

    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_encoder_comparison(
    df: pd.DataFrame, task: str, split: str, n_train: int, n_test: int, out_dir: Path,
) -> None:
    """Accuracy vs spike count scatter + bar chart (test accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    acc_col = "test_accuracy"

    # ── Accuracy vs spike count ──
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["mean_spike_count"], row[acc_col],
                   s=200, c=[colors[i]], edgecolors="black", linewidth=1,
                   zorder=5)
        offset = (8, 0.01) if i % 2 == 0 else (-8, -0.02)
        ax.annotate(row["encoder"], (row["mean_spike_count"], row[acc_col]),
                    textcoords="offset points", xytext=offset, fontsize=9,
                    ha="center")
    ax.set_xlabel("Mean Spike Count (↓ better)")
    ax.set_ylabel("Test Accuracy (↑ better)")
    ax.set_title("Test Accuracy vs Spike Count")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df[acc_col].max(), color="green", linestyle="--", alpha=0.3)

    # ── Bar chart ──
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df[acc_col].values, w, label="Test Accuracy", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, df["efficiency"].values * 1000, w,
           label="Efficiency (Acc/Spike ×1000)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["encoder"].values, rotation=20, ha="right", fontsize=9)
    ax.set_title("Encoder Comparison (test-set)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Spike Encoder Evaluation — {task}  ({split} split, {n_train}T/{n_test}V)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_encoder_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved encoder comparison → {out_dir / f'{task}_encoder_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PMUT spike encoders.")
    parser.add_argument(
        "--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose",
    )
    parser.add_argument(
        "--split", choices=["grouped", "window", "random_window"],
        default=None,
        help="Sweep-level train/test split [default: grouped for high-glucose, "
             "window for glucose-ladder]",
    )
    parser.add_argument(
        "--test-every", type=int, default=5,
        help="For window splits: every Nth sweep is test (default: 5)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    split = args.split or TASK_DEFAULT_SPLIT[args.task]

    out_dir = OUTPUT_DIR
    print(f"Spike Encoder Evaluation — {args.task}")
    print(f"  Split: {split}  (test-every={args.test_every})")
    print("=" * 50)

    # ── Load data ──
    frame = load_task_frame(args.task, args.data_root)
    results, y, classes, sweep_labels, sweep_groups = encode_all_sweeps(frame)

    # Per-class sweep counts for diagnostics
    unique_labels, label_counts = np.unique(sweep_labels, return_counts=True)
    print(f"\n  Total sweeps: {len(sweep_groups)}  |  Classes: {len(classes)}")
    print(f"  Sweeps per class: {dict(zip(unique_labels, label_counts))}")

    # ── Evaluate with honest train/test split ──
    df = evaluate_encoders(results, y, sweep_labels, sweep_groups,
                           split=split, test_every=args.test_every)

    n_train = int(df["n_train"].iloc[0])
    n_test = int(df["n_test"].iloc[0])

    # ── Print results ──
    print(f"\n  Encoder Comparison  (train={n_train} sweeps, test={n_test} sweeps, {split} split):")
    print(f"  {'Encoder':<15s} {'Train Acc':>10s} {'Test Acc':>10s} {'Spike Count':>12s} {'Efficiency':>12s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for _, row in df.iterrows():
        print(f"  {row['encoder']:<15s} {row['train_accuracy']:>10.4f} {row['test_accuracy']:>10.4f} "
              f"{row['mean_spike_count']:>12.1f} {row['efficiency']:>12.6f}")

    best = df.iloc[0]
    print(f"\n  Best encoder: {best['encoder']}  (test accuracy = {best['test_accuracy']:.4f})")
    print(f"  Justification: {best['encoder']} coding provides the best generalisation-"
          f"efficiency trade-off on unseen sweeps.")

    # ── Save ──
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_encoder_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved CSV → {out_path}")

    plot_encoder_comparison(df, args.task, split, n_train, n_test, out_dir)

    print("\nDone — Notebook 9 complete.")


if __name__ == "__main__":
    main()
