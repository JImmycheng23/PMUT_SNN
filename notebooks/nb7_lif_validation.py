"""
Notebook 7: LIF Neuron Refresher and Validation
================================================
Implement and validate a LIF neuron from scratch:
  - Current injection, threshold, reset, refractory period
  - Membrane potential traces
  - Spike raster plots
  - Firing rate vs input current curve
  - Parameter sweeps (threshold, time constant, refractory period)

Run:  python notebooks/nb7_lif_validation.py
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "lif_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LIFNeuron:
    """Discrete-time LIF neuron with refractory period.

    Dynamics:
        V[t] = beta * V[t-1] + I[t]          (leaky integration)
        if V[t] >= threshold:
            spike = 1
            V[t] = V_reset
            refractory_counter = tau_refractory
        refractory_counter > 0 → no integration, just countdown.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        beta: float = 0.9,
        v_reset: float = 0.0,
        tau_refractory: int = 2,
    ):
        self.threshold = threshold
        self.beta = beta
        self.v_reset = v_reset
        self.tau_refractory = tau_refractory
        self.v = 0.0
        self.refractory = 0

    def reset(self):
        self.v = 0.0
        self.refractory = 0

    def step(self, current: float) -> int:
        if self.refractory > 0:
            self.refractory -= 1
            self.v = self.v_reset
            return 0
        # leaky integration
        self.v = self.beta * self.v + current
        if self.v >= self.threshold:
            self.v = self.v_reset
            self.refractory = self.tau_refractory
            return 1
        return 0

    def run(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run neuron over a sequence of input currents."""
        T = len(currents)
        voltage = np.zeros(T)
        spikes = np.zeros(T, dtype=int)
        for t in range(T):
            voltage[t] = self.v
            spikes[t] = self.step(currents[t])
        return voltage, spikes


# ---------------------------------------------------------------------------
# Experiment 1: Single current pulse injection
# ---------------------------------------------------------------------------

def experiment_1_current_injection(out_dir: Path) -> None:
    """Inject a step current and observe membrane potential and spikes."""
    T = 200
    currents = np.zeros(T)
    currents[20:150] = 1.5  # above-threshold current

    neuron = LIFNeuron(threshold=1.0, beta=0.9, tau_refractory=3)
    voltage, spikes = neuron.run(currents)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(currents, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Input Current")
    axes[0].set_title("Step Current Injection")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(voltage, color="#2ca02c", linewidth=1.5)
    axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Threshold")
    axes[1].set_ylabel("Membrane Voltage")
    axes[1].set_title("Membrane Potential")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    spike_times = np.where(spikes)[0]
    axes[2].eventplot([spike_times], colors="#d62728", linewidths=1.5)
    axes[2].set_ylabel("Spike")
    axes[2].set_title("Spike Raster")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylim(0.5, 1.5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("LIF Neuron — Step Current Response", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_current_injection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp1_current_injection.png")


# ---------------------------------------------------------------------------
# Experiment 2: Firing rate vs input current (f-I curve)
# ---------------------------------------------------------------------------

def experiment_2_fi_curve(out_dir: Path) -> None:
    """Compute firing rate as a function of constant input current."""
    T = 500
    currents = np.linspace(0.0, 3.0, 60)
    rates = []
    neuron = LIFNeuron(threshold=1.0, beta=0.9, tau_refractory=2)

    for I in currents:
        neuron.reset()
        spikes = 0
        for t in range(T):
            # skip initial transient (first 50 steps)
            s = neuron.step(I)
            if t >= 50:
                spikes += s
        rate = spikes / (T - 50)
        rates.append(rate)

    # theoretical rate for LIF:  r = 1 / (tau_refractory - ln(1 - threshold/I) / ln(beta))
    # (simplified; we just plot empirical)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(currents, rates, "o-", color="#1f77b4", markersize=5, linewidth=1.5)
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Input Current")
    ax.set_ylabel("Firing Rate (spikes/step)")
    ax.set_title("LIF Neuron — Firing Rate vs Input Current (f-I curve)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_fi_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp2_fi_curve.png")


# ---------------------------------------------------------------------------
# Experiment 3: Parameter sweep
# ---------------------------------------------------------------------------

def experiment_3_parameter_sweep(out_dir: Path) -> None:
    """Show how beta (time constant), threshold, and refractory period affect firing."""
    T = 300
    I = 1.5  # constant above-threshold current

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Sweep beta (leak) ---
    ax = axes[0, 0]
    betas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for beta in betas:
        neuron = LIFNeuron(threshold=1.0, beta=beta, tau_refractory=2)
        voltage, _ = neuron.run(np.full(T, I))
        ax.plot(voltage[:100], linewidth=1.2, label=f"β={beta}")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Effect of β (Leak / Time Constant)")
    ax.set_ylabel("Membrane Voltage")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Sweep threshold ---
    ax = axes[0, 1]
    thresholds = [0.5, 0.75, 1.0, 1.25, 1.5]
    for th in thresholds:
        neuron = LIFNeuron(threshold=th, beta=0.9, tau_refractory=2)
        voltage, _ = neuron.run(np.full(T, I))
        ax.plot(voltage[:100], linewidth=1.2, label=f"θ={th}")
        ax.axhline(y=th, color="red", linestyle="--", alpha=0.2)
    ax.set_title("Effect of Threshold (θ)")
    ax.set_ylabel("Membrane Voltage")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Sweep refractory period ---
    ax = axes[1, 0]
    refs = [0, 2, 5, 10, 20]
    for ref in refs:
        neuron = LIFNeuron(threshold=1.0, beta=0.9, tau_refractory=ref)
        voltage, _ = neuron.run(np.full(T, I))
        ax.plot(voltage[:100], linewidth=1.2, label=f"τ_ref={ref}")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Effect of Refractory Period")
    ax.set_ylabel("Membrane Voltage")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Firing rate heatmap (beta vs threshold) ---
    ax = axes[1, 1]
    betas_2d = np.linspace(0.5, 0.99, 30)
    thresh_2d = np.linspace(0.3, 2.0, 30)
    rate_grid = np.zeros((len(thresh_2d), len(betas_2d)))
    for i, th in enumerate(thresh_2d):
        for j, beta in enumerate(betas_2d):
            neuron = LIFNeuron(threshold=th, beta=beta, tau_refractory=2)
            spk = 0
            for t in range(T):
                if t >= 50:
                    spk += neuron.step(I)
            rate_grid[i, j] = spk / (T - 50)

    im = ax.imshow(rate_grid, origin="lower", aspect="auto",
                   extent=[betas_2d[0], betas_2d[-1], thresh_2d[0], thresh_2d[-1]])
    ax.set_xlabel("β (leak)")
    ax.set_ylabel("Threshold θ")
    ax.set_title(f"Firing Rate (I={I}) — β vs θ")
    fig.colorbar(im, ax=ax, label="Firing Rate")

    fig.suptitle("LIF Neuron — Parameter Sweep", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_parameter_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp3_parameter_sweep.png")


# ---------------------------------------------------------------------------
# Experiment 4: Two connected LIF neurons
# ---------------------------------------------------------------------------

def experiment_4_two_neurons(out_dir: Path) -> None:
    """Two LIF neurons with an excitatory synapse."""
    T = 300
    I_pre = np.zeros(T)
    I_pre[20:200] = 1.5  # pre-synaptic neuron input

    pre = LIFNeuron(threshold=1.0, beta=0.9, tau_refractory=2)
    post = LIFNeuron(threshold=1.0, beta=0.92, tau_refractory=3)

    v_pre = np.zeros(T)
    v_post = np.zeros(T)
    s_pre = np.zeros(T, dtype=int)
    s_post = np.zeros(T, dtype=int)
    syn_current = np.zeros(T)

    syn_weight = 0.8  # excitatory synaptic weight
    syn_decay = 0.85

    for t in range(T):
        v_pre[t] = pre.v
        s_pre[t] = pre.step(I_pre[t])

        # synaptic current from pre-synaptic spike
        if s_pre[t]:
            syn_current[t] += syn_weight
        else:
            syn_current[t] = syn_current[t-1] * syn_decay if t > 0 else 0.0

        v_post[t] = post.v
        if post.refractory == 0:
            s_post[t] = post.step(syn_current[t])
        else:
            post.refractory -= 1
            post.v = 0.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(I_pre, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Input")
    axes[0].set_title("Pre-synaptic Input Current")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(v_pre, color="#2ca02c", linewidth=1.2)
    axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    sp1 = np.where(s_pre)[0]
    for t in sp1:
        axes[1].axvline(x=t, color="#d62728", alpha=0.3, linewidth=0.5)
    axes[1].set_ylabel("V_pre")
    axes[1].set_title("Pre-synaptic Membrane Potential")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(syn_current, color="#9467bd", linewidth=1.2)
    axes[2].set_ylabel("Synaptic Current")
    axes[2].set_title("Post-synaptic Current (EPSP)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(v_post, color="#ff7f0e", linewidth=1.2)
    axes[3].axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    sp2 = np.where(s_post)[0]
    for t in sp2:
        axes[3].axvline(x=t, color="#d62728", alpha=0.3, linewidth=0.5)
    axes[3].set_ylabel("V_post")
    axes[3].set_title("Post-synaptic Membrane Potential")
    axes[3].set_xlabel("Time step")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Two Connected LIF Neurons — Excitatory Synapse", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "exp4_two_neurons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp4_two_neurons.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = OUTPUT_DIR
    print("LIF Neuron Validation — Notebook 7")
    print("=" * 50)

    print("\nExperiment 1: Step Current Injection")
    experiment_1_current_injection(out_dir)

    print("\nExperiment 2: Firing Rate vs Input Current (f-I curve)")
    experiment_2_fi_curve(out_dir)

    print("\nExperiment 3: Parameter Sweep (β, θ, τ_refractory)")
    experiment_3_parameter_sweep(out_dir)

    print("\nExperiment 4: Two Connected LIF Neurons")
    experiment_4_two_neurons(out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done — Notebook 7 complete.")


if __name__ == "__main__":
    main()
