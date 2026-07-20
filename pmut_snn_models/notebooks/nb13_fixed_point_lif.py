"""
Notebook 13: Fixed-Point LIF Neuron
====================================
Convert the LIF neuron to fixed-point (Q-format) and compare with floating-point.

Experiments:
  1. Single LIF neuron: float vs fixed voltage traces
  2. Spike raster comparison: float vs fixed
  3. Bit-width sensitivity: 16/12/8/4-bit error analysis
  4. Quantization error propagation over time

Usage:
  python notebooks/nb13_fixed_point_lif.py
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.fixed_point import (
    QFormat, STANDARD_FORMATS, FloatLIF, FixedPointLIF,
    compare_lif_neurons, bit_width_sensitivity,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "fixed_point"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def experiment1_single_neuron_traces(out_dir: Path) -> None:
    """Compare voltage traces of float vs fixed LIF for one neuron."""
    print("\n[Experiment 1] Single LIF neuron: float vs fixed voltage traces")

    n_steps = 200
    n_neurons = 1
    n_input = 4
    seed = 42

    # Input: step currents of varying amplitude
    t = np.arange(n_steps)
    u = np.zeros((n_steps, n_input))
    u[20:60, 0] = 1.5    # Strong drive
    u[80:100, 1] = 0.8   # Sub-threshold
    u[120:160, 2] = 2.0   # Very strong
    u[170:190, 3] = 0.5   # Very weak

    float_lif = FloatLIF(n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)
    fixed_16 = FixedPointLIF(qfmt=STANDARD_FORMATS["16-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)
    fixed_12 = FixedPointLIF(qfmt=STANDARD_FORMATS["12-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)
    fixed_8 = FixedPointLIF(qfmt=STANDARD_FORMATS["8-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)

    models = [("Float (reference)", float_lif), ("16-bit Q3.12", fixed_16), ("12-bit Q3.8", fixed_12), ("8-bit Q3.4", fixed_8)]
    colors = ["black", "#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 10), sharex=True)

    for ax, (name, model), color in zip(axes, models, colors):
        model.reset()
        v_trace = np.zeros(n_steps)
        s_trace = np.zeros(n_steps)
        for t_idx in range(n_steps):
            spikes, v = model.step(u[t_idx : t_idx + 1])
            v_trace[t_idx] = v[0, 0]
            s_trace[t_idx] = spikes[0, 0]

        ax.plot(v_trace, color=color, linewidth=1.2, label="membrane potential")
        ax.set_ylabel(f"{name}\nVoltage", fontsize=9)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5, label="threshold")

        # Mark spikes
        spike_times = np.where(s_trace > 0.5)[0]
        for st in spike_times:
            ax.axvline(x=st, color="red", alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Fixed-Point LIF Neuron: Voltage Traces at Different Bit Widths", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "exp1_voltage_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def experiment2_spike_raster_comparison(out_dir: Path) -> None:
    """Compare spike rasters across bit widths for a population of neurons."""
    print("\n[Experiment 2] Spike raster comparison across bit widths")

    n_steps = 100
    n_neurons = 20
    n_input = 4
    seed = 123

    rng = np.random.RandomState(seed)
    u = rng.randn(n_steps, n_input).astype(np.float64) * 0.8

    float_lif = FloatLIF(n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)
    configs = [
        ("16-bit Q3.12", FixedPointLIF(qfmt=STANDARD_FORMATS["16-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#2ecc71"),
        ("12-bit Q3.8", FixedPointLIF(qfmt=STANDARD_FORMATS["12-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#f39c12"),
        ("8-bit Q3.4", FixedPointLIF(qfmt=STANDARD_FORMATS["8-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#e74c3c"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Float reference (top)
    float_lif.reset()
    float_spikes = np.zeros((n_steps, n_neurons))
    for t_idx in range(n_steps):
        s, _ = float_lif.step(u[t_idx : t_idx + 1])
        float_spikes[t_idx] = s[0]
    spike_t, spike_n = np.where(float_spikes > 0.5)
    axes[0].scatter(spike_t, spike_n, s=8, c="black", marker="|", alpha=0.7)
    axes[0].set_ylabel("Float (ref)", fontsize=9)

    for ax, (name, model, color) in zip(axes[1:], configs):
        model.reset()
        spikes_arr = np.zeros((n_steps, n_neurons))
        for t_idx in range(n_steps):
            s, _ = model.step(u[t_idx : t_idx + 1])
            spikes_arr[t_idx] = s[0]
        st, sn = np.where(spikes_arr > 0.5)
        ax.scatter(st, sn, s=8, c=color, marker="|", alpha=0.7)

        # Highlight mismatches
        mismatch = (float_spikes != spikes_arr).astype(float)
        mt, mn = np.where(mismatch > 0.5)
        if len(mt) > 0:
            ax.scatter(mt, mn, s=12, c="red", marker="x", alpha=0.5, linewidths=0.5,
                      label=f"{len(mt)} mismatches / {int(float_spikes.sum())} ref spikes")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel(name, fontsize=9)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Fixed-Point LIF Neuron: Spike Raster Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "exp2_spike_rasters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def experiment3_bit_width_sensitivity(out_dir: Path) -> None:
    """Quantify error vs bit width."""
    print("\n[Experiment 3] Bit-width sensitivity analysis")

    n_steps = 200
    n_neurons = 50
    n_input = 4
    seed = 99

    rng = np.random.RandomState(seed)
    u = rng.randn(n_steps, n_input).astype(np.float64) * 0.8

    results = bit_width_sensitivity(u, n_steps=n_steps, n_neurons=n_neurons, n_input=n_input, seed=seed)

    # Table
    print(f"\n  {'Format':<20s} {'Bits':>6s} {'V_MAE':>10s} {'V_RMSE':>10s} {'Spike Mismatch':>15s} {'Timing Err':>12s}")
    print(f"  {'-'*75}")
    for r in results:
        print(f"  {r['qfmt_name']:<20s} {r['bit_width']:>6d} {r['v_mae']:>10.6f} {r['v_rmse']:>10.6f} {r['spike_mismatch_rate']:>14.4f}  {r['spike_timing_error']:>10.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    bits = [r["bit_width"] for r in results]
    names = [r["qfmt_name"] for r in results]

    # V MAE
    axes[0, 0].bar(names, [r["v_mae"] for r in results], color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"], edgecolor="white")
    axes[0, 0].set_ylabel("Membrane Voltage MAE")
    axes[0, 0].set_title("Voltage MAE vs Bit Width")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Spike mismatch
    axes[0, 1].bar(names, [r["spike_mismatch_rate"] for r in results], color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"], edgecolor="white")
    axes[0, 1].set_ylabel("Spike Mismatch Rate")
    axes[0, 1].set_title("Spike Mismatch vs Bit Width")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Timing error
    axes[1, 0].bar(names, [r["spike_timing_error"] for r in results], color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"], edgecolor="white")
    axes[1, 0].set_ylabel("Mean Spike Timing Error (steps)")
    axes[1, 0].set_title("Timing Error vs Bit Width")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # V RMSE
    axes[1, 1].bar(names, [r["v_rmse"] for r in results], color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"], edgecolor="white")
    axes[1, 1].set_ylabel("Membrane Voltage RMSE")
    axes[1, 1].set_title("Voltage RMSE vs Bit Width")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("Fixed-Point LIF: Bit-Width Sensitivity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "exp3_bit_width_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def experiment4_error_accumulation(out_dir: Path) -> None:
    """Track how quantization error accumulates over time."""
    print("\n[Experiment 4] Quantization error accumulation over time")

    n_steps = 300
    n_neurons = 1
    n_input = 4
    seed = 77

    rng = np.random.RandomState(seed)
    u = rng.randn(n_steps, n_input).astype(np.float64) * 0.6

    float_lif = FloatLIF(n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed)
    configs = [
        ("16-bit Q3.12", FixedPointLIF(qfmt=STANDARD_FORMATS["16-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#2ecc71"),
        ("12-bit Q3.8", FixedPointLIF(qfmt=STANDARD_FORMATS["12-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#f39c12"),
        ("8-bit Q3.4", FixedPointLIF(qfmt=STANDARD_FORMATS["8-bit"], n_input=n_input, n_neurons=n_neurons, beta=0.9, threshold=1.0, seed=seed), "#e74c3c"),
    ]

    float_lif.reset()
    float_v = np.zeros(n_steps)
    fixed_v_all = {}

    for name, model, _ in configs:
        model.reset()
        fixed_v_all[name] = np.zeros(n_steps)

    for t_idx in range(n_steps):
        _, fv = float_lif.step(u[t_idx : t_idx + 1])
        float_v[t_idx] = fv[0, 0]
        for name, model, _ in configs:
            _, qv = model.step(u[t_idx : t_idx + 1])
            fixed_v_all[name][t_idx] = qv[0, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Voltage traces
    axes[0].plot(float_v, color="black", linewidth=1.5, label="Float (reference)", alpha=0.8)
    for name, _, color in configs:
        axes[0].plot(fixed_v_all[name], color=color, linewidth=1.0, label=name, alpha=0.7)
    axes[0].set_ylabel("Membrane Voltage")
    axes[0].set_title("Voltage Trace Over Time")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Cumulative error
    for name, _, color in configs:
        error = np.abs(float_v - fixed_v_all[name])
        cum_error = np.cumsum(error)
        axes[1].plot(cum_error, color=color, linewidth=1.5, label=name)

    axes[1].set_ylabel("Cumulative |Error|")
    axes[1].set_xlabel("Time Step")
    axes[1].set_title("Cumulative Absolute Voltage Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Fixed-Point LIF: Error Accumulation Over Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "exp4_error_accumulation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main() -> None:
    print(f"{'='*60}")
    print("NB13: Fixed-Point LIF Neuron")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment1_single_neuron_traces(out_dir)
    experiment2_spike_raster_comparison(out_dir)
    experiment3_bit_width_sensitivity(out_dir)
    experiment4_error_accumulation(out_dir)

    print(f"\n{'='*60}")
    print("NB13 complete")
    print(f"  Outputs → {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
