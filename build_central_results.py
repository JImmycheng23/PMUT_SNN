"""Build central results master CSV/JSON from all experiment outputs.

Fixed per Review Point 10 (2026-07-24):
  - All seed directories scanned (not just _seed7)
  - Proper LSM path globs (lsm/ subdirectory)
  - Separate numeric columns: accuracy_mean, accuracy_std, num_seeds
  - All model types collected: neural, ESN, LSM membrane, LSM spikes,
    encoder-comparison, fixed-point, feature ML, HDL status
  - Each row: task, model, input_representation, prediction_level,
    fold, seed, accuracy, accuracy_std, num_seeds, balanced_accuracy,
    macro_f1, per_class_recall, spike_count, bit_width, result_file_path
"""

import csv
import json
import os
import subprocess
from pathlib import Path

ROWS: list[dict] = []
OUTPUT_DIR = Path("outputs/central_results")


def add_row(**kw) -> None:
    ROWS.append(kw)


def _try_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Neural models (train.py)
# ═══════════════════════════════════════════════════════════════════════════

def collect_neural_results() -> None:
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        return

    # Walk recursively: supports both flat (old) and nested fold-based (new) layouts
    # New: outputs/runs/{task}_folds_w{win}/{model}/fold{N}/[seed dirs...]
    # Old: outputs/runs/{task}_grouped_{model}_w{win}_seed{N}/

    # ── Single-seed metrics.json (any depth) ──
    for mf in sorted(runs_dir.glob("**/metrics.json")):
        # Skip metrics_summary.json files caught by the glob
        if mf.name != "metrics.json":
            continue
        d = _try_json(mf)
        if d is None:
            continue

        model = d.get("model", "?")
        ds_meta = d.get("dataset", {})
        task = ds_meta.get("task", "?")
        split = ds_meta.get("split", "?")
        fold = ds_meta.get("fold", None)
        window = ds_meta.get("window_size", None)
        enc = ds_meta.get("encoder", None)
        enc_chan = ds_meta.get("encode_channels", "")

        if enc:
            input_repr = f"spike_encoded({enc},{enc_chan})"
        elif "spike" in model or "lif" in model:
            input_repr = "continuous_learned_spiking"
        else:
            input_repr = "continuous"

        add_row(
            experiment="neural",
            task=task,
            split=str(split),
            fold=fold,
            window=window,
            model=model,
            input_representation=input_repr,
            prediction_level="window",
            seed=d.get("seed"),
            num_seeds=1,
            accuracy=d.get("final_test_accuracy"),
            accuracy_std=None,
            balanced_accuracy=d.get("balanced_accuracy"),
            macro_f1=d.get("macro_f1"),
            best_val_accuracy=d.get("best", {}).get("val_accuracy"),
            best_epoch=d.get("best", {}).get("epoch"),
            per_class_recall=json.dumps(d.get("per_class_recall", {})),
            spike_count=None,
            bit_width=None,
            hdl_verified=None,
            result_file_path=str(mf.relative_to("outputs")),
            git_commit=_git_commit(),
        )

    # ── Multi-seed summaries (metrics_summary.json at any depth) ──
    for sp in sorted(runs_dir.glob("**/metrics_summary.json")):
        d = _try_json(sp)
        if d is None or "final_test_accuracy_mean" not in d:
            continue
        ds_meta = d.get("dataset", {})
        # Extract model name from path: .../{model}/fold{N}/metrics_summary.json
        model = d.get("model", sp.parent.parent.name if sp.parent.name.startswith("fold") else "?")
        fold = ds_meta.get("fold", None)
        if fold is None and sp.parent.name.startswith("fold"):
            try:
                fold = int(sp.parent.name.replace("fold", ""))
            except ValueError:
                pass
        add_row(
            experiment="neural_summary",
            task=ds_meta.get("task", "?"),
            split=str(ds_meta.get("split", "?")),
            fold=fold,
            window=ds_meta.get("window_size", None),
            model=model,
            input_representation="various",
            prediction_level="window",
            seed=None,
            num_seeds=d.get("num_seeds"),
            accuracy=d.get("final_test_accuracy_mean"),
            accuracy_std=d.get("final_test_accuracy_std"),
            balanced_accuracy=None,
            macro_f1=None,
            best_val_accuracy=d.get("best_val_accuracy_mean"),
            best_epoch=None,
            per_class_recall=None,
            spike_count=None,
            bit_width=None,
            hdl_verified=None,
            result_file_path=str(sp.relative_to("outputs")),
            git_commit=_git_commit(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Reservoir — ESN + LSM membrane + LSM spikes (NB11)
# ═══════════════════════════════════════════════════════════════════════════

def collect_reservoir_results() -> None:
    res_dir = Path("outputs/reservoir")
    if not res_dir.exists():
        return

    # New layout: outputs/reservoir/{task}/{split}/fold{N}/[lsm/]{metrics}.json
    # Walk recursively for all metric files
    for mf in sorted(res_dir.glob("**/metrics_summary.json")):
        _add_reservoir_row(mf, is_summary=True)

    for mf in sorted(res_dir.glob("**/metrics.json")):
        _add_reservoir_row(mf, is_summary=False)

    for mf in sorted(res_dir.glob("**/metrics_spikes.json")):
        d = _try_json(mf)
        is_summ = d is not None and "test_accuracy_mean" in d
        _add_reservoir_row(mf, is_summary=is_summ, force_model="LSM (spikes)")


def _add_reservoir_row(mf: Path, is_summary: bool, force_model: str | None = None) -> None:
    """Parse one reservoir metrics file and add a row."""
    d = _try_json(mf)
    if d is None:
        return

    # Infer model type and state_mode from path
    parts = mf.parts
    # Path: outputs/reservoir/{task}/{split}/fold{N}/[lsm/]{file}
    task = parts[2] if len(parts) > 2 else "?"
    split_name = parts[3] if len(parts) > 3 else "?"
    fold_dir = parts[4] if len(parts) > 4 else "?"

    # Extract fold number
    fold = None
    if fold_dir.startswith("fold"):
        try:
            fold = int(fold_dir.replace("fold", ""))
        except ValueError:
            pass

    window = None
    for p in split_name.split("_w"):
        pass
    wp = split_name.split("_w")
    if len(wp) == 2:
        try:
            window = int(wp[1])
        except ValueError:
            pass

    # Determine model
    if force_model:
        model = force_model
    elif "lsm" in str(mf).lower():
        model = "LSM (membrane)"
    else:
        model = "ESN (tanh)"

    state_mode = "tanh" if "ESN" in model else "spikes" if "spikes" in str(mf).lower() else "membrane"
    input_repr = f"continuous_reservoir_{state_mode}"

    if is_summary and "test_accuracy_mean" in d:
        add_row(
            experiment="reservoir_summary",
            task=task, split=split_name, fold=fold, window=window,
            model=model, input_representation=input_repr,
            prediction_level="window",
            seed=None, num_seeds=d.get("num_seeds"),
            accuracy=d.get("test_accuracy_mean"),
            accuracy_std=d.get("test_accuracy_std"),
            balanced_accuracy=None, macro_f1=None,
            best_val_accuracy=None, best_epoch=None,
            per_class_recall=None, spike_count=None, bit_width=None, hdl_verified=None,
            result_file_path=str(mf.relative_to("outputs")),
            git_commit=_git_commit(),
        )
    elif not is_summary:
        params = d.get("params", {})
        add_row(
            experiment="reservoir",
            task=task, split=split_name, fold=fold, window=window,
            model=model, input_representation=input_repr,
            prediction_level="window",
            seed=params.get("seed", d.get("seed")), num_seeds=1,
            accuracy=d.get("test_accuracy"), accuracy_std=None,
            balanced_accuracy=d.get("balanced_accuracy"), macro_f1=None,
            best_val_accuracy=None, best_epoch=None,
            per_class_recall=json.dumps(d.get("per_class_recall", {})),
            spike_count=None, bit_width=None, hdl_verified=None,
            result_file_path=str(mf.relative_to("outputs")),
            git_commit=_git_commit(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Encoder comparison (NB9)
# ═══════════════════════════════════════════════════════════════════════════

def collect_encoder_results() -> None:
    enc_dir = Path("outputs/encoder_evaluation")
    if not enc_dir.exists():
        return

    for csv_path in sorted(enc_dir.glob("*_encoder_comparison.csv")):
        task = csv_path.stem.replace("_encoder_comparison", "")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for _, row in df.iterrows():
            encoder = row.get("encoder", "?")
            acc_col = None
            for c in ["test_accuracy", "accuracy", "score"]:
                if c in row.index and pd.notna(row[c]):
                    acc_col = c
                    break
            if acc_col is None:
                continue

            add_row(
                experiment="encoder_comparison",
                task=task,
                split="grouped",
                fold=None,
                window=None,
                model=f"Encoder({encoder})+LR",
                input_representation=f"spike_aggregate({encoder})",
                prediction_level="sweep",
                seed=None,
                num_seeds=1,
                accuracy=float(row[acc_col]),
                accuracy_std=None,
                balanced_accuracy=None,
                macro_f1=None,
                best_val_accuracy=None,
                best_epoch=None,
                per_class_recall=None,
                spike_count=None,
                bit_width=None,
                hdl_verified=None,
                result_file_path=str(csv_path.relative_to("outputs")),
                git_commit=_git_commit(),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Fixed-point PMUT (NB14)
# ═══════════════════════════════════════════════════════════════════════════

def collect_fixed_point_results() -> None:
    fp_dir = Path("outputs/fixed_point")
    if not fp_dir.exists():
        return

    for task_dir in sorted(fp_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        bf = task_dir / "bit_width_study.json"
        d = _try_json(bf)
        if d is None:
            continue

        entries = d if isinstance(d, list) else [d]
        for entry in entries:
            fmt = entry.get("format", entry.get("qfmt_name", "?"))
            acc = entry.get("accuracy", entry.get("accuracy_quantized"))
            bw = entry.get("bit_width", None)
            if bw is None and "bit" in str(fmt):
                try:
                    bw = int(str(fmt).split("-bit")[0].strip())
                except (ValueError, IndexError):
                    pass

            add_row(
                experiment="fixed_point",
                task=task,
                split="grouped",
                fold=None,
                window=None,
                model="LSM (quantized)",
                input_representation="continuous_quantized",
                prediction_level="window",
                seed=None,
                num_seeds=1,
                accuracy=acc,
                accuracy_std=None,
                balanced_accuracy=None,
                macro_f1=None,
                best_val_accuracy=None,
                best_epoch=None,
                per_class_recall=None,
                spike_count=None,
                bit_width=bw,
                hdl_verified=None,
                result_file_path=str(bf.relative_to("outputs")),
                git_commit=_git_commit(),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Feature ML baselines (NB5)
# ═══════════════════════════════════════════════════════════════════════════

def collect_feature_ml_results() -> None:
    ml_dir = Path("outputs/feature_ml")
    if not ml_dir.exists():
        return

    for csv_path in sorted(ml_dir.glob("*_baseline_summary.csv")):
        task = csv_path.stem.replace("_baseline_summary", "")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for _, row in df.iterrows():
            model = row.get("model", "?")
            acc = row.get("accuracy")
            f1 = row.get("f1_macro")

            add_row(
                experiment="feature_ml",
                task=task,
                split="CV",
                fold=None,
                window=None,
                model=str(model),
                input_representation="sweep_features",
                prediction_level="sweep",
                seed=None,
                num_seeds=1,
                accuracy=float(acc) if acc is not None else None,
                accuracy_std=None,
                balanced_accuracy=None,
                macro_f1=float(f1) if f1 is not None else None,
                best_val_accuracy=None,
                best_epoch=None,
                per_class_recall=None,
                spike_count=None,
                bit_width=None,
                hdl_verified=None,
                result_file_path=str(csv_path.relative_to("outputs")),
                git_commit=_git_commit(),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 6. HDL verification status
# ═══════════════════════════════════════════════════════════════════════════

def collect_hdl_status() -> None:
    hdl_dir = Path("outputs/hdl")
    tv_dir = Path("outputs/hdl_test_vectors")

    checks = [
        ("HDL specification", hdl_dir / "HDL_SPECIFICATION.md"),
        ("Verilog LIF neuron", hdl_dir / "lif_neuron_fxp.v"),
        ("Test vectors (16-bit)", tv_dir / "high-glucose" / "16-bit" / "params.json"),
        ("Test vectors (12-bit)", tv_dir / "high-glucose" / "12-bit" / "params.json"),
    ]

    for label, path in checks:
        add_row(
            experiment="hdl_status",
            task="high-glucose",
            split=None,
            fold=None,
            window=None,
            model=label,
            input_representation=None,
            prediction_level=None,
            seed=None,
            num_seeds=None,
            accuracy=None,
            accuracy_std=None,
            balanced_accuracy=None,
            macro_f1=None,
            best_val_accuracy=None,
            best_epoch=None,
            per_class_recall=None,
            spike_count=None,
            bit_width=None,
            hdl_verified=path.exists(),
            result_file_path=str(path) if path.exists() else None,
            git_commit=_git_commit(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    collect_neural_results()
    collect_reservoir_results()
    collect_encoder_results()
    collect_fixed_point_results()
    collect_feature_ml_results()
    collect_hdl_status()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "task",
        "split",
        "fold",
        "window",
        "model",
        "input_representation",
        "prediction_level",
        "seed",
        "num_seeds",
        "accuracy",
        "accuracy_std",
        "balanced_accuracy",
        "macro_f1",
        "best_val_accuracy",
        "best_epoch",
        "per_class_recall",
        "spike_count",
        "bit_width",
        "hdl_verified",
        "result_file_path",
        "git_commit",
    ]

    csv_path = OUTPUT_DIR / "master_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ROWS)

    json_path = OUTPUT_DIR / "master_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ROWS, f, indent=2, default=str)

    print(f"Wrote {len(ROWS)} rows to {csv_path}")
    print(f"Wrote {len(ROWS)} rows to {json_path}")

    from collections import Counter
    by_exp = Counter(r["experiment"] for r in ROWS)
    print("\nRows per experiment type:")
    for exp, count in by_exp.most_common():
        print(f"  {exp}: {count}")


if __name__ == "__main__":
    main()
