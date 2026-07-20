"""
Notebook 5: Feature-Based ML Baselines
=======================================
Train classical ML models on extracted PMUT physical features.
Models: Logistic Regression, SVM, Random Forest, KNN, MLP.

Run:  python notebooks/nb5_feature_ml_baselines.py [--task pure]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    LeaveOneOut,
    cross_val_score,
    cross_val_predict,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.data import _label_sort_key

FEATURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "features"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "feature_ml"

# Features to use (avoid derived/baseline features to prevent leakage)
FEATURE_COLS = [
    "f0_hz", "amp_peak", "phase_at_resonance_deg",
    "bandwidth_hz", "q_factor", "phase_slope_deg_per_bin",
    "amp_mean", "amp_std", "amp_auc",
    "x_mean", "y_mean", "noise_floor",
]

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42, probability=True),
    "SVM (Linear)": SVC(kernel="linear", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
}


def load_features(task: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{task}_feature_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run nb4_feature_extraction.py --task {task} first.\nMissing: {path}")
    return pd.read_csv(path)


def prepare_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Note: {len(missing)} features unavailable: {missing}")
    X = df[available].copy()
    # Fill NaN with column median
    X = X.fillna(X.median())
    y_raw = df["label"].values
    classes = sorted(set(y_raw), key=_label_sort_key)
    y = np.array([classes.index(lb) for lb in y_raw])
    return X.values.astype(np.float64), y, classes


def plot_confusion_matrices(
    results: dict, classes: list[str], task: str, out_dir: Path
) -> None:
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.atleast_2d(axes)
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // cols][idx % cols]
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(f"{name}\nAcc={res['accuracy']:.3f}  F1={res['f1_macro']:.3f}", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    # hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle(f"Feature-Based ML — Confusion Matrices ({task})", fontsize=12, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices → {out_dir / f'{task}_confusion_matrices.png'}")


def plot_accuracy_comparison(results: dict, task: str, out_dir: Path) -> None:
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1, w, label="Macro F1", color="#ff7f0e", alpha=0.85)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Feature-Based ML — Model Comparison ({task})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved accuracy comparison → {out_dir / f'{task}_accuracy_comparison.png'}")


def run_baselines(task: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Feature-Based ML Baselines — {task}")
    print(f"{'='*60}")

    df = load_features(task)
    X, y, classes = prepare_X_y(df)
    print(f"  Samples: {len(y)} | Classes: {len(classes)} | Features: {X.shape[1]}")
    print(f"  Classes: {classes}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation — pick a strategy that works for the sample count.
    # StratifiedKFold fails when any class has < n_splits samples (common here).
    min_per_class = int(np.min(np.bincount(y)))
    n_splits = min(5, min_per_class)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_name = f"{n_splits}-fold stratified"
    elif min_per_class >= 2:
        # KFold still respects class balance better than LOO for N >= 2
        cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
        cv_name = f"{min(5, len(y))}-fold (unstratified, min class size={min_per_class})"
    else:
        cv = LeaveOneOut()
        cv_name = "leave-one-out"

    print(f"  CV strategy: {cv_name}")

    results = {}
    for name, model in MODELS.items():
        # --- honest cross-validated metrics ---
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        # Cross-validated predictions for confusion matrix + F1
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
        cv_acc = float(cv_scores.mean())
        cv_f1 = float(f1_score(y, y_pred_cv, average="macro"))
        cm = confusion_matrix(y, y_pred_cv)

        results[name] = {
            "accuracy": cv_acc,
            "f1_macro": cv_f1,
            "cv_mean": cv_acc,
            "cv_std": float(cv_scores.std()),
            "cm": cm,
        }
        print(f"  {name:<22s}  CV-Acc={cv_acc:.4f}  CV-F1={cv_f1:.4f}  (±{cv_scores.std():.4f})")

    out_dir = OUTPUT_DIR
    plot_confusion_matrices(results, classes, task, out_dir)
    plot_accuracy_comparison(results, task, out_dir)

    # Save summary CSV
    summary = []
    for name, res in results.items():
        summary.append({
            "model": name,
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "cv_mean": res["cv_mean"],
            "cv_std": res["cv_std"],
        })
    pd.DataFrame(summary).to_csv(out_dir / f"{task}_baseline_summary.csv", index=False)
    print(f"  Saved summary → {out_dir / f'{task}_baseline_summary.csv'}")

    # RF feature importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)
    available = [c for c in FEATURE_COLS if c in df.columns]
    importances = pd.DataFrame({
        "feature": available,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances["feature"], importances["importance"], color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest Feature Importance ({task})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance → {out_dir / f'{task}_feature_importance.png'}")
    print("\n  Top 5 features:")
    for _, row in importances.head(5).iterrows():
        print(f"    {row['feature']:<30s} {row['importance']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train feature-based ML baselines on PMUT data.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    args = parser.parse_args()
    run_baselines(args.task)
    print("\nDone — Notebook 5 complete.")


if __name__ == "__main__":
    main()
