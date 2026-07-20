"""
Notebook 11: PMUT Reservoir Computing
======================================
Treat each PMUT frequency sweep as a temporal signal and classify it using
an Echo State Network (ESN) reservoir model.

The sweep is processed sequentially:
  frequency point 1 → frequency point 2 → ... → frequency point N

A reservoir transforms the input sequence into a high-dimensional state, and
a trained linear readout (ridge regression) performs classification.

Usage:
  # Single run with default hyperparameters
  python notebooks/nb11_reservoir.py --task pure --split random_window --window-size 128

  # Hyperparameter sweep
  python notebooks/nb11_reservoir.py --task pure --split random_window --sweep

  # Memory capacity estimation
  python notebooks/nb11_reservoir.py --memory-capacity

  # Reservoir state dynamics analysis
  python notebooks/nb11_reservoir.py --task pure --split random_window --analyze
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from src.pmut_models.data import (
    build_dataset,
    load_task_frame,
    make_sequence_samples,
    split_indices,
    standardize_from_train,
    encode_labels,
    _label_sort_key,
    SIGNAL_COLUMNS,
)
from src.pmut_models.reservoir import (
    Reservoir,
    reservoir_grid_search,
    estimate_memory_capacity,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reservoir"

# ---------------------------------------------------------------------------
# Default hyperparameter grid for sweep
# ---------------------------------------------------------------------------

DEFAULT_PARAM_GRID = {
    "n_reservoir": [100, 200, 300, 500],
    "spectral_radius": [0.8, 0.9, 0.95, 0.99],
    "leaking_rate": [0.1, 0.3, 0.5, 0.9],
    "input_scaling": [0.1, 0.5, 1.0],
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reservoir(
    bundle,
    n_reservoir: int = 300,
    spectral_radius: float = 0.95,
    leaking_rate: float = 0.3,
    input_scaling: float = 1.0,
    sparsity: float = 0.9,
    warmup: int = 10,
    state_reduction: str = "mean",
    alpha: float = 1.0,
    seed: int = 42,
) -> dict:
    """Train an ESN reservoir on a DatasetBundle and return metrics."""
    n_input = len(bundle.channels)

    reservoir = Reservoir(
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leaking_rate=leaking_rate,
        input_scaling=input_scaling,
        seed=seed,
    )

    train_acc = reservoir.fit_readout(
        bundle.x_train, bundle.y_train,
        warmup=warmup,
        state_reduction=state_reduction,
        alpha=alpha,
    )

    preds = reservoir.predict(
        bundle.x_test,
        warmup=warmup,
        state_reduction=state_reduction,
    )

    test_acc = float((preds == bundle.y_test).mean())

    # Per-class accuracy
    per_class = {}
    for i, label in enumerate(bundle.labels):
        mask = bundle.y_test == i
        if mask.any():
            per_class[label] = float((preds[mask] == bundle.y_test[mask]).mean())

    # Confusion matrix
    n_classes = len(bundle.labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(bundle.y_test.tolist(), preds.tolist()):
        cm[true, pred] += 1

    return {
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "targets": bundle.y_test.tolist(),
        "labels": bundle.labels,
        "params": {
            "n_reservoir": n_reservoir,
            "spectral_radius": spectral_radius,
            "leaking_rate": leaking_rate,
            "input_scaling": input_scaling,
            "sparsity": sparsity,
            "warmup": warmup,
            "state_reduction": state_reduction,
            "alpha": alpha,
        },
    }


# ---------------------------------------------------------------------------
# Multi-seed training
# ---------------------------------------------------------------------------

def train_reservoir_multi_seed(bundle, num_seeds: int = 5, base_seed: int = 7, **kwargs) -> dict:
    """Train reservoir with multiple seeds, return mean ± std metrics."""
    results = []
    for i in range(num_seeds):
        seed = base_seed + i * 37
        r = train_reservoir(bundle, seed=seed, **kwargs)
        results.append(r)

    accs = [r["test_accuracy"] for r in results]
    summary = {
        "num_seeds": num_seeds,
        "test_accuracy_mean": float(np.mean(accs)),
        "test_accuracy_std": float(np.std(accs)),
        "test_accuracy_list": accs,
        "individual_results": results,
        "labels": bundle.labels,
        "params": results[0]["params"] if results else {},
    }
    return summary


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], task: str, out_dir: Path) -> Path:
    """Plot confusion matrix."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Annotate
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Reservoir Confusion Matrix — {task}")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reservoir_states(
    states_by_class: dict,
    task: str,
    out_dir: Path,
    max_dim: int = 3,
) -> Path:
    """PCA of reservoir states colored by class."""
    from sklearn.decomposition import PCA

    all_labels = []
    all_states = []
    for label, state_list in states_by_class.items():
        for s in state_list:
            # s may be (seq_len, n_reservoir) — reduce to mean over time for PCA
            if s.ndim == 2:
                s = s.mean(axis=0)
            all_states.append(s)
            all_labels.append(label)

    if not all_states:
        return None

    X = np.stack(all_states)
    pca = PCA(n_components=max_dim)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, max(2, max_dim), figsize=(5 * max_dim, 4))
    if max_dim <= 2:
        axes = [axes]

    unique_labels = sorted(set(all_labels), key=_label_sort_key)
    cmap = plt.cm.tab10
    color_map = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    # PC1 vs PC2
    ax = axes[0]
    for label in unique_labels:
        mask = np.array([l == label for l in all_labels])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color_map[label]],
                   label=label, alpha=0.7, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"Reservoir State PCA — {task}")
    ax.legend(fontsize=7, loc="best")

    # PC1 vs PC3 (if 3+ components)
    if max_dim >= 3:
        ax = axes[1]
        for label in unique_labels:
            mask = np.array([l == label for l in all_labels])
            ax.scatter(X_pca[mask, 0], X_pca[mask, 2], c=[color_map[label]],
                       label=label, alpha=0.7, s=20)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title(f"Reservoir State PCA — {task}")
        ax.legend(fontsize=7, loc="best")

    # Explained variance bar
    ax = axes[-1]
    ax.bar(range(1, max_dim + 1), pca.explained_variance_ratio_[:max_dim])
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")

    fig.suptitle(f"Reservoir State Dynamics — {task}", fontsize=14, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_state_pca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sweep_results(results: list[dict], task: str, out_dir: Path) -> Path:
    """Plot hyperparameter sweep results."""
    valid = [r for r in results if r.get("test_acc") is not None]
    if not valid:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Group by each parameter and find best per value
    params_to_plot = ["n_reservoir", "spectral_radius", "leaking_rate", "input_scaling"]
    for ax, param in zip(axes.flat, params_to_plot):
        # Aggregate: for each value of this param, take mean test acc
        by_val = {}
        for r in valid:
            val = r["params"].get(param)
            if val is None:
                continue
            by_val.setdefault(val, []).append(r["test_acc"])

        vals = sorted(by_val.keys())
        means = [np.mean(by_val[v]) for v in vals]
        stds = [np.std(by_val[v]) for v in vals]

        ax.errorbar(vals, means, yerr=stds, marker="o", capsize=4, linewidth=2)
        ax.set_xlabel(param)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Accuracy vs {param}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Reservoir Hyperparameter Sweep — {task}", fontsize=14, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_memory_capacity(mc_result: dict, out_dir: Path) -> Path:
    """Plot memory capacity curve."""
    delays = list(range(1, mc_result["max_delay"] + 1))
    mc = mc_result["mc_per_delay"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(delays, mc, width=0.8, color="steelblue", edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Delay k")
    ax.set_ylabel("MC(k)")
    ax.set_title(f"Memory Capacity (total MC = {mc_result['total_mc']:.2f})")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "memory_capacity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_class_accuracy(per_class: dict, task: str, out_dir: Path) -> Path:
    """Plot per-class accuracy bar chart."""
    labels = sorted(per_class.keys(), key=_label_sort_key)
    accs = [per_class[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    colors = ["#2ecc71" if a >= 0.7 else "#f39c12" if a >= 0.4 else "#e74c3c" for a in accs]
    ax.bar(range(len(labels)), accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Reservoir Per-Class Accuracy — {task}")
    ax.grid(axis="y", alpha=0.3)

    # Annotate
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_per_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Reservoir state analysis
# ---------------------------------------------------------------------------

def analyze_reservoir_states(bundle, reservoir_params: dict, task: str, out_dir: Path) -> dict:
    """Collect and analyze reservoir state trajectories per class."""
    reservoir = Reservoir(
        n_input=len(bundle.channels),
        **reservoir_params,
    )

    states_by_class = {}
    separability = {}

    for i, label in enumerate(bundle.labels):
        # Get one test sample per class
        mask = bundle.y_test == i
        if not mask.any():
            continue
        idx = np.where(mask)[0][0]
        sample = bundle.x_test[idx]

        states = reservoir.get_state_trajectory(sample, warmup=10)
        states_by_class[label] = [states]

    # Compute pairwise linear separability
    from sklearn.svm import LinearSVC

    for i, label_a in enumerate(bundle.labels):
        for j, label_b in enumerate(bundle.labels):
            if i >= j:
                continue
            mask_a = bundle.y_test == i
            mask_b = bundle.y_test == j
            if not mask_a.any() or not mask_b.any():
                continue

            # Collect states for all samples of these two classes
            X_pair = []
            y_pair = []
            for idx in np.where(mask_a)[0]:
                s = reservoir.get_state_trajectory(bundle.x_test[idx], warmup=10).mean(axis=0)
                X_pair.append(s)
                y_pair.append(0)
            for idx in np.where(mask_b)[0]:
                s = reservoir.get_state_trajectory(bundle.x_test[idx], warmup=10).mean(axis=0)
                X_pair.append(s)
                y_pair.append(1)

            X_pair = np.stack(X_pair)
            y_pair = np.array(y_pair)

            try:
                clf = LinearSVC(C=1.0, max_iter=2000, dual=False)
                clf.fit(X_pair, y_pair)
                sep_score = float(clf.score(X_pair, y_pair))
            except Exception:
                sep_score = 0.5

            separability[f"{label_a} vs {label_b}"] = sep_score

    # Plot
    plot_path = plot_reservoir_states(states_by_class, task, out_dir)

    return {"states_captured": len(states_by_class), "separability": separability, "pca_plot": str(plot_path)}


# ---------------------------------------------------------------------------
# Modality comparison
# ---------------------------------------------------------------------------

def run_modality_comparison(bundle, reservoir_params: dict, task: str, out_dir: Path) -> list[dict]:
    """Compare reservoir accuracy with different input channel subsets."""
    # Map channel subsets
    channel_subsets = {
        "amplitude_only": [0, 2],        # amplitude, x_v
        "phase_only": [1, 3],            # phase, y_v
        "amplitude_phase": [0, 1, 2, 3], # all 4 channels
    }

    results = []
    for name, channels in channel_subsets.items():
        try:
            n_input = len(channels)
            reservoir = Reservoir(
                n_input=n_input,
                **reservoir_params,
            )

            x_train_sub = bundle.x_train[:, :, channels]
            x_test_sub = bundle.x_test[:, :, channels]

            train_acc = reservoir.fit_readout(x_train_sub, bundle.y_train, warmup=10, state_reduction="mean")
            preds = reservoir.predict(x_test_sub, warmup=10, state_reduction="mean")
            test_acc = float((preds == bundle.y_test).mean())

            results.append({
                "modality": name,
                "channels": channels,
                "n_input": n_input,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            })
        except Exception as e:
            results.append({"modality": name, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PMUT Reservoir Computing — Notebook 11")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--split", choices=["window", "random_window", "grouped"], default="random_window")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)

    # Reservoir hyperparameters
    parser.add_argument("--n-reservoir", type=int, default=300)
    parser.add_argument("--spectral-radius", type=float, default=0.95)
    parser.add_argument("--leaking-rate", type=float, default=0.3)
    parser.add_argument("--input-scaling", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--state-reduction", choices=["mean", "last", "concat_ends"], default="mean")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--num-seeds", type=int, default=1)

    # Modes
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter grid search")
    parser.add_argument("--analyze", action="store_true", help="Run reservoir state dynamics analysis")
    parser.add_argument("--modality", action="store_true", help="Run amplitude vs phase comparison")
    parser.add_argument("--memory-capacity", action="store_true", help="Estimate memory capacity")
    parser.add_argument("--all", action="store_true", help="Run all analyses (sweep + analyze + modality + memory)")

    args = parser.parse_args()

    out_dir = OUTPUT_DIR / args.task / f"{args.split}_w{args.window_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_sweep = args.sweep or args.all
    run_analyze = args.analyze or args.all
    run_modality = args.modality or args.all
    run_memory = args.memory_capacity or args.all

    print(f"{'='*60}")
    print(f"PMUT Reservoir Computing — Notebook 11")
    print(f"Task: {args.task}  Split: {args.split}  Window: {args.window_size}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset...")
    bundle = build_dataset(
        task=args.task,
        split=args.split,
        data_root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"  Train: {bundle.x_train.shape}  Test: {bundle.x_test.shape}")
    print(f"  Classes: {bundle.labels}")
    print(f"  Unique groups train/test: {len(set(bundle.train_groups))}/{len(set(bundle.test_groups))}")

    # ------------------------------------------------------------------
    # 2. Memory capacity estimation (independent of task data)
    # ------------------------------------------------------------------
    if run_memory:
        print("\n[2/5] Estimating memory capacity...")
        mc = estimate_memory_capacity(n_reservoir=args.n_reservoir, max_delay=50)
        mc_path = out_dir / "memory_capacity.json"
        with open(mc_path, "w") as f:
            json.dump(mc, f, indent=2)
        print(f"  Total MC: {mc['total_mc']:.2f} (max delay = {mc['max_delay']})")
        plot_memory_capacity(mc, out_dir)

    # ------------------------------------------------------------------
    # 3. Hyperparameter sweep
    # ------------------------------------------------------------------
    best_params = {
        "n_reservoir": args.n_reservoir,
        "spectral_radius": args.spectral_radius,
        "leaking_rate": args.leaking_rate,
        "input_scaling": args.input_scaling,
        "sparsity": args.sparsity,
    }

    if run_sweep:
        print("\n[2/5] Running hyperparameter sweep...")
        sweep_results = reservoir_grid_search(
            bundle.x_train, bundle.y_train,
            bundle.x_test, bundle.y_test,
            param_grid=DEFAULT_PARAM_GRID,
            n_input=len(bundle.channels),
            warmup=args.warmup,
            state_reduction=args.state_reduction,
        )

        # Save sweep results
        sweep_path = out_dir / "hyperparameter_sweep.csv"
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["n_reservoir", "spectral_radius", "leaking_rate",
                                                    "input_scaling", "train_acc", "test_acc"])
            writer.writeheader()
            for r in sweep_results:
                if r.get("test_acc") is not None:
                    writer.writerow({
                        "n_reservoir": r["params"].get("n_reservoir"),
                        "spectral_radius": r["params"].get("spectral_radius"),
                        "leaking_rate": r["params"].get("leaking_rate"),
                        "input_scaling": r["params"].get("input_scaling"),
                        "train_acc": f"{r['train_acc']:.4f}",
                        "test_acc": f"{r['test_acc']:.4f}",
                    })

        # Best result
        best = sweep_results[0] if sweep_results else {}
        if best.get("test_acc") is not None:
            print(f"  Best sweep config: {best['params']}")
            print(f"  Best sweep test accuracy: {best['test_acc']:.4f}")
            best_params.update(best["params"])

        plot_sweep_results(sweep_results, args.task, out_dir)
        print(f"  Sweep results saved → {sweep_path}")

        # Also save full JSON
        with open(out_dir / "hyperparameter_sweep.json", "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 4. Train reservoir with best/default params
    # ------------------------------------------------------------------
    print(f"\n[{'3' if run_sweep else '2'}/5] Training reservoir...")
    print(f"  Params: n={best_params['n_reservoir']}, SR={best_params['spectral_radius']}, "
          f"LR={best_params['leaking_rate']}, IS={best_params['input_scaling']}")

    if args.num_seeds > 1:
        summary = train_reservoir_multi_seed(
            bundle, num_seeds=args.num_seeds,
            n_reservoir=best_params["n_reservoir"],
            spectral_radius=best_params["spectral_radius"],
            leaking_rate=best_params["leaking_rate"],
            input_scaling=best_params["input_scaling"],
            sparsity=best_params["sparsity"],
            warmup=args.warmup,
            state_reduction=args.state_reduction,
            alpha=args.alpha,
        )
        test_acc = summary["test_accuracy_mean"]
        test_std = summary["test_accuracy_std"]
        print(f"  Test accuracy: {test_acc:.4f} ± {test_std:.4f} ({args.num_seeds} seeds)")

        with open(out_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Use first individual result for detailed analysis
        result = summary["individual_results"][0]
    else:
        result = train_reservoir(
            bundle,
            n_reservoir=best_params["n_reservoir"],
            spectral_radius=best_params["spectral_radius"],
            leaking_rate=best_params["leaking_rate"],
            input_scaling=best_params["input_scaling"],
            sparsity=best_params["sparsity"],
            warmup=args.warmup,
            state_reduction=args.state_reduction,
            alpha=args.alpha,
            seed=42,
        )
        test_acc = result["test_accuracy"]
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for label, acc in sorted(result["per_class_accuracy"].items(), key=lambda x: _label_sort_key(x[0])):
        bar = "█" * int(acc * 20)
        print(f"    {label:<25s}: {acc:.4f} {bar}")

    # Plot confusion matrix
    cm_path = plot_confusion_matrix(
        np.array(result["confusion_matrix"]), result["labels"], args.task, out_dir
    )
    print(f"\n  Confusion matrix → {cm_path}")

    # Plot per-class accuracy
    pc_path = plot_per_class_accuracy(result["per_class_accuracy"], args.task, out_dir)

    # Save predictions CSV
    preds_path = out_dir / "predictions.csv"
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "prediction", "target_label", "prediction_label"])
        writer.writeheader()
        for t, p in zip(result["targets"], result["predictions"]):
            writer.writerow({
                "target": t,
                "prediction": p,
                "target_label": result["labels"][int(t)],
                "prediction_label": result["labels"][int(p)],
            })

    # ------------------------------------------------------------------
    # 5. Modality comparison
    # ------------------------------------------------------------------
    if run_modality:
        step = "4" if run_sweep else "3"
        print(f"\n[{step}/5] Running modality comparison (amplitude vs phase vs combined)...")
        modality_results = run_modality_comparison(bundle, best_params, args.task, out_dir)

        with open(out_dir / "modality_comparison.json", "w") as f:
            json.dump(modality_results, f, indent=2)

        print(f"  {'Modality':<20s} {'Test Acc':>10s}")
        print(f"  {'-'*30}")
        for mr in modality_results:
            if "error" in mr:
                print(f"  {mr['modality']:<20s} ERROR: {mr['error']}")
            else:
                print(f"  {mr['modality']:<20s} {mr['test_accuracy']:>10.4f}")

        # Modality bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [m["modality"] for m in modality_results if "error" not in m]
        accs = [m["test_accuracy"] for m in modality_results if "error" not in m]
        ax.bar(names, accs, color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="white")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Input Modality Comparison — {args.task}")
        ax.set_ylim(0, 1.05)
        for i, a in enumerate(accs):
            ax.text(i, a + 0.02, f"{a:.4f}", ha="center", fontweight="bold")
        fig.tight_layout()
        mod_path = out_dir / f"{args.task}_modality.png"
        fig.savefig(mod_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Modality plot → {mod_path}")

    # ------------------------------------------------------------------
    # 6. Reservoir state analysis
    # ------------------------------------------------------------------
    if run_analyze:
        step = "4" if run_sweep else "3"
        step = "5" if run_sweep else step
        print(f"\n[{step}/5] Analyzing reservoir state dynamics...")
        analysis = analyze_reservoir_states(bundle, best_params, args.task, out_dir)

        with open(out_dir / "state_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"  States captured for {analysis['states_captured']}/{len(bundle.labels)} classes")
        print(f"  Pairwise separability:")
        for pair, score in sorted(analysis["separability"].items()):
            print(f"    {pair:<35s}: {score:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Notebook 11 complete — {args.task}")
    print(f"  Test accuracy: {test_acc:.4f}" + (f" ± {test_std:.4f}" if args.num_seeds > 1 else ""))
    print(f"  Outputs → {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
