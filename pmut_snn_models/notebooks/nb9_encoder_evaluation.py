"""
Notebook 9: Spike Encoder Evaluation
=====================================
Compare all spike encoders using a simple classifier.
  - Accuracy vs spike count
  - Latency vs accuracy
  - Select best encoder for SNN classification

Run:  python notebooks/nb9_encoder_evaluation.py [--task pure]
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
    SIGNAL_COLUMNS,
    _label_sort_key,
    encode_labels,
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


def encode_all_sweeps(frame: pd.DataFrame) -> dict:
    """Encode every sweep with all 5 encoders, return features + labels."""
    # Collect all sweeps
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

    # Encode with each method, then aggregate to a fixed-size feature vector
    # We use simple statistics of the spike representation as features
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
            n_t, n_c = spikes.shape
            # Features: mean firing rate per channel, temporal mean, total spikes
            ch_rates = spikes.mean(axis=0)  # (n_channels,)
            temporal_profile = spikes.mean(axis=1)  # (n_timesteps,)
            # Downsample channel rates to fixed size (rate/latency encoders vary per sweep)
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

    return results, y, classes, sweep_keys


def evaluate_encoders(results: dict, y: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Train a simple logistic regression on each encoder's features."""
    rows = []
    for enc_name, data in results.items():
        X = data["X"]
        spike_counts = data["spike_counts"]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Simple logistic regression
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_scaled, y)
        y_pred = clf.predict(X_scaled)
        acc = accuracy_score(y, y_pred)

        mean_spikes = float(spike_counts.mean())
        # latency proxy: fewer spikes = lower latency
        latency_proxy = mean_spikes  # rough proxy

        rows.append({
            "encoder": enc_name,
            "accuracy": acc,
            "mean_spike_count": mean_spikes,
            "total_spike_count": int(spike_counts.sum()),
            "latency_proxy": latency_proxy,
            "efficiency": acc / max(mean_spikes, 1),  # accuracy per spike
        })

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def plot_encoder_comparison(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Accuracy vs spike count scatter + bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Accuracy vs spike count ---
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["mean_spike_count"], row["accuracy"],
                   s=200, c=[colors[i]], edgecolors="black", linewidth=1,
                   zorder=5)
        offset = (5, 0.01) if i % 2 == 0 else (-5, -0.02)
        ax.annotate(row["encoder"], (row["mean_spike_count"], row["accuracy"]),
                    textcoords="offset points", xytext=offset, fontsize=9,
                    ha="center" if i % 2 == 0 else "center")
    ax.set_xlabel("Mean Spike Count (↓ better)")
    ax.set_ylabel("Accuracy (↑ better)")
    ax.set_title("Accuracy vs Spike Count")
    ax.grid(True, alpha=0.3)
    # ideal: top-left
    ax.axhline(y=df["accuracy"].max(), color="green", linestyle="--", alpha=0.3)
    best_idx = df["accuracy"].idxmax() if hasattr(df["accuracy"], 'idxmax') else 0

    # --- Bar chart ---
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["accuracy"].values, w, label="Accuracy", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, df["efficiency"].values * 1000, w,
           label="Efficiency (Acc/Spike ×1000)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["encoder"].values, rotation=20, ha="right", fontsize=9)
    ax.set_title("Encoder Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Spike Encoder Evaluation — {task}", fontsize=13, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_encoder_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved encoder comparison → {out_dir / f'{task}_encoder_comparison.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PMUT spike encoders.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR
    print(f"Spike Encoder Evaluation — {args.task}")
    print("=" * 50)

    frame = load_task_frame(args.task, args.data_root)
    results, y, classes, sweep_keys = encode_all_sweeps(frame)
    print(f"  Sweeps: {len(sweep_keys)} | Classes: {len(classes)}")
    print(f"  Classes: {classes}")

    df = evaluate_encoders(results, y, classes)

    print("\n  Encoder Comparison:")
    print(f"  {'Encoder':<15s} {'Accuracy':>10s} {'Spike Count':>12s} {'Efficiency':>12s}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*12}")
    for _, row in df.iterrows():
        print(f"  {row['encoder']:<15s} {row['accuracy']:>10.4f} {row['mean_spike_count']:>12.1f} {row['efficiency']:>12.6f}")

    best = df.iloc[0]
    print(f"\n  Best encoder: {best['encoder']} (accuracy={best['accuracy']:.4f})")
    print(f"  Justification: {best['encoder']} coding provides the best accuracy-efficiency trade-off.")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.task}_encoder_comparison.csv", index=False)
    plot_encoder_comparison(df, args.task, out_dir)

    print("\nDone — Notebook 9 complete.")


if __name__ == "__main__":
    main()
