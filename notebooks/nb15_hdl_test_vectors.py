"""
Notebook 15: HDL Test-Vector Generation
========================================
Generate PMUT-derived test vectors for HDL LIF neuron verification.

Produces:
  pmut_input_spikes.csv    — spike input to HDL
  expected_voltage_q.csv   — expected Q-format membrane voltage
  expected_spikes.csv      — expected spike output
  params.json              — threshold, beta, bit-width, Q-format info

Usage:
  python notebooks/nb15_hdl_test_vectors.py [--task pure]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.data import build_dataset
from src.pmut_models.fixed_point import (
    QFormat, STANDARD_FORMATS, generate_hdl_test_vectors, save_hdl_test_vectors,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "hdl_test_vectors"


def extract_pmut_spike_example(bundle, neuron_idx: int = 0, n_steps: int = 50) -> np.ndarray:
    """Extract a representative PMUT sweep segment as input spikes.

    Takes one test sample and converts amplitude channels to spike-like inputs
    via simple threshold-crossing (simulates spike-encoded PMUT data).
    """
    # Use first training sample (NOT test — demo only, but keep test set untouched)
    sample = bundle.x_train[0]  # (seq_len, n_channels)

    # Convert amplitude channels (0, 2) to binary spike trains via threshold-crossing
    seq_len = min(n_steps, sample.shape[0])
    n_input = sample.shape[1]

    spikes = np.zeros((seq_len, n_input), dtype=np.int8)

    for ch in range(n_input):
        signal = sample[:seq_len, ch]
        # Threshold-crossing encoding: spike on significant change
        diff = np.abs(np.diff(signal, prepend=signal[0]))
        threshold = diff.std() * 0.5
        spikes[1:, ch] = (diff[1:] > threshold).astype(np.int8)

    return spikes


def main() -> None:
    parser = argparse.ArgumentParser(description="HDL Test-Vector Generation — NB15")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--split", default="grouped")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--n-neurons", type=int, default=10, help="Number of LIF neurons for HDL testbench")
    parser.add_argument("--n-steps", type=int, default=64, help="Number of time steps for test vectors")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"NB15: HDL Test-Vector Generation")
    print(f"Task: {args.task}  Neurons: {args.n_neurons}  Steps: {args.n_steps}")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1] Loading PMUT dataset...")
    bundle = build_dataset(task=args.task, split=args.split, window_size=args.window_size)
    print(f"  Test samples: {bundle.x_test.shape[0]}")

    # Extract PMUT spike example
    print("\n[2] Extracting PMUT spike-encoded input...")
    input_spikes = extract_pmut_spike_example(bundle, n_steps=args.n_steps)
    total_input_spikes = int(input_spikes.sum())
    print(f"  Input shape: {input_spikes.shape}  Total spikes: {total_input_spikes}")

    # Generate test vectors at multiple bit widths
    print(f"\n[3] Generating HDL test vectors...")

    for name, qfmt in [("16-bit", STANDARD_FORMATS["16-bit"]), ("12-bit", STANDARD_FORMATS["12-bit"])]:
        print(f"\n  --- {name} ({qfmt}) ---")

        test_data = generate_hdl_test_vectors(
            input_spikes=input_spikes,
            beta=0.9,
            threshold=1.0,
            qfmt=qfmt,
            n_neurons=args.n_neurons,
            seed=42,
        )

        bit_dir = out_dir / name.replace(" ", "_")
        save_hdl_test_vectors(test_data, bit_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"NB15 complete")
    print(f"  HDL test vectors ready for verification")
    print(f"\n  File descriptions:")
    print(f"    pmut_input_spikes.csv   — PMUT-derived binary spike input")
    print(f"    expected_voltage_q.csv  — Expected membrane voltage (Q-format integers)")
    print(f"    expected_spikes.csv     — Expected spike output (binary)")
    print(f"    params.json             — Neuron parameters + Q-format configuration")
    print(f"  Outputs → {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
