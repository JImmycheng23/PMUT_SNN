"""
Notebook 8: PMUT Spike Encoding
================================
Convert PMUT frequency sweeps into spike trains using 5 encoding methods:
  1. Rate coding — amplitude → Poisson firing rate
  2. Latency coding — amplitude → first-spike delay
  3. Threshold-crossing coding — spike when signal crosses a level
  4. Delta/event coding — spike on significant change
  5. Population coding — multiple neurons tuned to different frequency bands

⚠️  Architectural note — this is an EXPLICIT (hand-designed) spike encoder.
It is NOT used by the trainable SNN models (spiking.py / train.py). Those
models take continuous-valued spectra as input and learn their own internal
spike representations via surrogate-gradient LIF neurons. The two pipelines
are independent and complementary:
  - NB8/NB9: "What hand-designed encoding best preserves class information?"
  - train.py: "Can a learned spiking network classify PMUT sweeps end-to-end?"

Run:  python notebooks/nb8_spike_encoding.py [--task high-glucose]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import load_task_frame, SIGNAL_COLUMNS, _label_sort_key

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "spike_encoding"


# ---------------------------------------------------------------------------
# Encoder 1: Rate coding
# ---------------------------------------------------------------------------

def rate_encode(
    signal: np.ndarray,
    n_steps: int = 100,
    max_rate: float = 0.5,
    min_val: float | None = None,
    max_val: float | None = None,
) -> np.ndarray:
    """Convert amplitude values to Poisson spike trains.

    Returns: (n_steps, len(signal)) binary array.
    Higher amplitude → more spikes per time step.
    """
    if min_val is None:
        min_val = float(signal.min())
    if max_val is None:
        max_val = float(signal.max())
    norm = np.clip((signal - min_val) / max(max_val - min_val, 1e-10), 0.0, 1.0)
    rates = norm * max_rate
    spikes = np.random.rand(n_steps, len(signal)) < rates[np.newaxis, :]
    return spikes.astype(np.int8)


# ---------------------------------------------------------------------------
# Encoder 2: Latency coding (time-to-first-spike)
# ---------------------------------------------------------------------------

def latency_encode(
    signal: np.ndarray,
    max_time: int = 100,
    min_val: float | None = None,
    max_val: float | None = None,
) -> np.ndarray:
    """Convert amplitude to first-spike latency.

    Higher amplitude → earlier spike.
    Returns: (max_time, len(signal)) binary array with exactly 1 spike per channel.
    """
    if min_val is None:
        min_val = float(signal.min())
    if max_val is None:
        max_val = float(signal.max())
    norm = np.clip((signal - min_val) / max(max_val - min_val, 1e-10), 0.01, 1.0)
    spike_time = ((1.0 - norm) * (max_time - 1)).astype(int)
    spike_time = np.clip(spike_time, 0, max_time - 1)
    spikes = np.zeros((max_time, len(signal)), dtype=np.int8)
    spikes[spike_time, np.arange(len(signal))] = 1
    return spikes


# ---------------------------------------------------------------------------
# Encoder 3: Threshold-crossing coding
# ---------------------------------------------------------------------------

def threshold_crossing_encode(
    signal: np.ndarray,
    n_thresholds: int = 8,
) -> np.ndarray:
    """Spike when signal crosses evenly-spaced amplitude thresholds.

    Returns: (len(signal)-1, n_thresholds) binary array.
    Each threshold level is a separate spike channel.
    """
    min_val = float(signal.min())
    max_val = float(signal.max())
    thresholds = np.linspace(min_val + 0.05 * (max_val - min_val),
                             max_val - 0.05 * (max_val - min_val), n_thresholds)
    T = len(signal)
    spikes = np.zeros((T - 1, n_thresholds), dtype=np.int8)
    for j, th in enumerate(thresholds):
        for i in range(T - 1):
            # spike on upward crossing
            if signal[i] < th and signal[i + 1] >= th:
                spikes[i, j] = 1
    return spikes


# ---------------------------------------------------------------------------
# Encoder 4: Delta / event coding
# ---------------------------------------------------------------------------

def delta_encode(
    signal: np.ndarray,
    sensitivity: float = 1.0,
    n_channels: int = 2,
) -> np.ndarray:
    """Spike on significant amplitude change.

    Returns: (len(signal)-1, n_channels) binary array.
    Channel 0: positive delta, Channel 1: negative delta.
    """
    diff = np.diff(signal)
    std = np.std(diff)
    if std < 1e-12:
        std = 1.0
    threshold = sensitivity * std
    T = len(signal) - 1
    spikes = np.zeros((T, n_channels), dtype=np.int8)
    # ON channel: positive changes
    spikes[diff > threshold, 0] = 1
    # OFF channel: negative changes
    spikes[diff < -threshold, min(1, n_channels - 1)] = 1
    return spikes


# ---------------------------------------------------------------------------
# Encoder 5: Population / Gaussian receptive field coding
# ---------------------------------------------------------------------------

def population_encode(
    signal: np.ndarray,
    n_neurons: int = 20,
    n_steps: int = 50,
) -> np.ndarray:
    """Population coding over frequency bands.

    Each neuron has a Gaussian tuning curve centered at a different
    frequency band. The entire sweep is treated as one time step,
    but each neuron's activation is rate-coded over n_steps.

    Returns: (n_steps, n_neurons) binary array.
    """
    n_freq = len(signal)
    centers = np.linspace(0, n_freq - 1, n_neurons)
    sigma = (n_freq / n_neurons) * 1.5

    activations = np.zeros(n_neurons)
    for j in range(n_neurons):
        # Gaussian weighting around center frequency
        weights = np.exp(-0.5 * ((np.arange(n_freq) - centers[j]) / sigma) ** 2)
        activations[j] = np.sum(np.abs(signal) * weights)

    # normalize and rate-code
    amax = activations.max()
    if amax > 1e-10:
        activations = activations / amax
    rates = activations * 0.3  # max 30% probability per step
    spikes = np.random.rand(n_steps, n_neurons) < rates[np.newaxis, :]
    return spikes.astype(np.int8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_spike_rasters(
    examples: dict,
    task: str,
    out_dir: Path,
) -> None:
    """Plot spike rasters for one example sweep per class."""
    classes = sorted(examples.keys(), key=_label_sort_key)
    encoders = ["rate", "latency", "threshold", "delta", "population"]

    fig, axes = plt.subplots(len(encoders), len(classes), figsize=(3 * len(classes), 2.5 * len(encoders)))

    for row, enc_name in enumerate(encoders):
        for col, cls in enumerate(classes):
            ax = axes[row][col] if len(classes) > 1 else axes[row]
            spikes = examples[cls].get(enc_name)
            if spikes is None or not spikes.any():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(f"{cls}\n{enc_name}", fontsize=8)
                continue
            # For visualization, show first 100 time steps, first 50 channels
            vis = spikes[:min(100, spikes.shape[0]), :min(50, spikes.shape[1])]
            spike_t, spike_ch = np.where(vis)
            if len(spike_t) > 5000:
                # subsample for performance
                idx = np.random.choice(len(spike_t), 5000, replace=False)
                spike_t, spike_ch = spike_t[idx], spike_ch[idx]
            ax.scatter(spike_t, spike_ch, s=1, c="black", marker="|", alpha=0.7)
            ax.set_xlim(0, min(100, spikes.shape[0]))
            ax.set_ylim(-0.5, min(50, spikes.shape[1]) - 0.5)
            n_spikes = int(spikes.sum())
            ax.set_title(f"{cls}\n{enc_name} ({n_spikes} spikes)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"PMUT Spike Encoding — Encoder Comparison ({task})", fontsize=13, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_spike_rasters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spike rasters → {out_dir / f'{task}_spike_rasters.png'}")


def compute_spike_stats(examples: dict) -> pd.DataFrame:
    """Compute spike count and sparsity per encoder per class."""
    rows = []
    for cls, encoders in examples.items():
        for enc_name, spikes in encoders.items():
            if spikes is None:
                continue
            n_spikes = int(spikes.sum())
            total = spikes.size
            rows.append({
                "class": cls,
                "encoder": enc_name,
                "n_spikes": n_spikes,
                "total_cells": total,
                "sparsity": 1.0 - n_spikes / total,
                "spike_density": n_spikes / total,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Encode one sweep
# ---------------------------------------------------------------------------

def encode_sweep(amp: np.ndarray, phase: np.ndarray) -> dict:
    """Encode a single sweep with all 5 methods.
    Uses both amplitude (primary) and phase (for threshold encoding).
    """
    # normalize amp for consistency
    amp_norm = (amp - amp.mean()) / max(amp.std(), 1e-10)

    return {
        "rate": rate_encode(amp, n_steps=100, max_rate=0.3),
        "latency": latency_encode(amp, max_time=100),
        "threshold": threshold_crossing_encode(amp, n_thresholds=8),
        "delta": delta_encode(amp, sensitivity=1.5, n_channels=2),
        "population": population_encode(amp, n_neurons=20, n_steps=50),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Encode PMUT sweeps as spike trains.")
    parser.add_argument("--task", choices=["high-glucose"], default="high-glucose",
        help="Primary task: high-glucose (5-class, 50–1000 mM, 3 sweeps/class). "
             "glucose-ladder retained in data.py for exploratory visualisation only.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PMUT Spike Encoding — {args.task}")
    print("=" * 50)

    frame = load_task_frame(args.task, args.data_root)
    labels = sorted(frame["label"].unique(), key=_label_sort_key)

    # Encode one example per class
    examples = {}
    for label in labels:
        subset = frame[frame["label"] == label]
        sweep_key = subset["sweep_key"].iloc[0]
        data = subset[subset["sweep_key"] == sweep_key].sort_values("frequency_hz")
        amp = data["amplitude"].to_numpy(dtype=float)
        phase = data["phase"].to_numpy(dtype=float)
        examples[label] = encode_sweep(amp, phase)
        total_spikes = sum(int(s.sum()) for s in examples[label].values())
        print(f"  {label:<20s}: {total_spikes} total spikes across all encoders")

    # Stats
    stats = compute_spike_stats(examples)
    stats_path = out_dir / f"{args.task}_spike_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"\n  Spike statistics saved → {stats_path}")
    print(stats.to_string(index=False))

    # Plot
    plot_spike_rasters(examples, args.task, out_dir)

    print("\nDone — Notebook 8 complete.")


if __name__ == "__main__":
    main()
