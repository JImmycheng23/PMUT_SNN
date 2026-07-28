"""
Reproducible full regeneration script for the PMUT-SNN project.

Runs every pipeline stage in order from a single frozen commit, producing
all datasets, features, model outputs, and the central results table.

Usage:
  python regenerate_all.py [--task high-glucose] [--skip-neural] [--skip-reservoir]

Outputs go to outputs/ exactly as described in REPRODUCIBILITY.md.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NEURAL_MODELS = ["cnn1d", "spiking_cnn1d", "temporal_lif_snn", "rate_snn_mlp"]
RESERVOIR_MODELS = ["esn", "lsm"]


def run(cmd: list[str], desc: str, timeout: int | None = None) -> int:
    """Run a command and print its output. Returns exit code."""
    print(f"\n{'=' * 72}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 72}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  ✓ OK  ({elapsed:.1f}s)")
    else:
        print(f"  ✗ FAILED (exit {result.returncode})  ({elapsed:.1f}s)")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible full regeneration")
    parser.add_argument("--task", default="high-glucose",
                        choices=["high-glucose"],
                        help="Classification task")
    parser.add_argument("--window-size", type=int, default=128,
                        help="Window size for sequence samples")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seeds per fold")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs for neural models")
    parser.add_argument("--skip-neural", action="store_true",
                        help="Skip neural model training (train.py)")
    parser.add_argument("--skip-reservoir", action="store_true",
                        help="Skip reservoir training (nb11_reservoir.py)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction + ML baselines")
    args = parser.parse_args()

    python = sys.executable
    task = args.task
    ws = str(args.window_size)
    seeds = str(args.num_seeds)
    epochs = str(args.epochs)

    failures: list[str] = []

    # ── Phase A: Prepare dataset ──
    rc = run(
        [python, "prepare_dataset.py", "--task", task, "--split", "grouped"],
        "Phase A: Prepare dataset",
    )
    if rc != 0:
        failures.append("prepare_dataset")

    # ── Phase B: Feature extraction (NB4) ──
    if not args.skip_features:
        rc = run(
            [python, "notebooks/nb4_feature_extraction.py", "--task", task],
            "Phase B: Feature extraction (NB4)",
        )
        if rc != 0:
            failures.append("nb4_feature_extraction")

        # ── Phase C: Feature ML baselines (NB5) ──
        rc = run(
            [python, "notebooks/nb5_feature_ml_baselines.py", "--task", task],
            "Phase C: Feature ML baselines (NB5)",
        )
        if rc != 0:
            failures.append("nb5_feature_ml_baselines")

    # ── Phase D: Neural models (train.py --folds) ──
    if not args.skip_neural:
        for model in NEURAL_MODELS:
            rc = run(
                [
                    python, "train.py",
                    "--task", task, "--split", "grouped", "--folds",
                    "--model", model, "--num-seeds", seeds,
                    "--epochs", epochs, "--window-size", ws,
                ],
                f"Phase D: Neural model — {model} ({seeds} seeds × 3 folds)",
                timeout=7200,  # 2 h per model
            )
            if rc != 0:
                failures.append(f"train/{model}")

    # ── Phase E: Reservoir models (nb11_reservoir.py --folds) ──
    if not args.skip_reservoir:
        # ESN
        rc = run(
            [
                python, "notebooks/nb11_reservoir.py",
                "--task", task, "--split", "grouped", "--folds",
                "--num-seeds", seeds, "--window-size", ws,
                "--state-mode", "membrane",
            ],
            f"Phase E: Reservoir — ESN ({seeds} seeds × 3 folds)",
        )
        if rc != 0:
            failures.append("reservoir/esn")

        # LSM
        rc = run(
            [
                python, "notebooks/nb11_reservoir.py",
                "--task", task, "--split", "grouped", "--folds",
                "--num-seeds", seeds, "--window-size", ws,
                "--lsm", "--state-mode", "membrane",
            ],
            f"Phase E: Reservoir — LSM ({seeds} seeds × 3 folds)",
        )
        if rc != 0:
            failures.append("reservoir/lsm")

    # ── Phase F: Model comparison (NB12) ──
    rc = run(
        [python, "notebooks/nb12_model_comparison.py", "--task", task],
        "Phase F: Model comparison (NB12)",
    )
    if rc != 0:
        failures.append("nb12_model_comparison")

    # ── Phase G: Central results ──
    rc = run(
        [python, "build_central_results.py"],
        "Phase G: Build central results table",
    )
    if rc != 0:
        failures.append("build_central_results")

    # ── Final ──
    print(f"\n{'=' * 72}")
    if failures:
        print(f"  COMPLETED WITH {len(failures)} FAILURES:")
        for f in failures:
            print(f"    ✗ {f}")
        print(f"  Re-run failed stages individually.")
        sys.exit(1)
    else:
        print(f"  ALL STAGES COMPLETED SUCCESSFULLY")
        print(f"  Outputs → {ROOT / 'outputs'}")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
