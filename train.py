import argparse
from pathlib import Path

import numpy as np

from src.pmut_models.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from src.pmut_models.data import build_dataset, build_dataset_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PMUT SNN/CNN benchmark models.")
    parser.add_argument("--task", choices=["high-glucose"], default="high-glucose",
        help="Primary task: high-glucose (5-class, 50–1000 mM, 3 sweeps/class). "
             "glucose-ladder retained in data.py for exploratory visualisation only.")
    parser.add_argument("--split", choices=["window", "random_window", "grouped"], default="window")
    parser.add_argument(
        "--model",
        choices=["rate_snn_mlp", "temporal_lif_snn", "spiking_cnn1d", "cnn1d", "encoded_spike_snn"],
        default="rate_snn_mlp",
    )
    parser.add_argument(
        "--encode",
        choices=["rate", "latency", "threshold", "delta", "population"],
        default=None,
        help="Pre-encode windows with an NB8 spike encoder before training. "
             "Requires --model encoded_spike_snn. "
             "This implements the intended pipeline: PMUT → encoder → SNN.",
    )
    parser.add_argument(
        "--encode-steps", type=int, default=50,
        help="Time steps for spike encoder (default: 50)",
    )
    parser.add_argument(
        "--encode-channels", choices=["amplitude", "phase", "amplitude+phase"],
        default="amplitude",
        help="Signal channel(s) to encode (default: amplitude). "
             "amplitude+phase encodes both independently and concatenates.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None, help="Window stride (default: window_size//2 for 50% overlap)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds for mean/std reporting (default: 5)")
    parser.add_argument("--folds", action="store_true",
        help="Use 3-fold sweep-level rotation (build_dataset_folds). "
             "Required for reproducible fold-based results. "
             "Without this flag, uses legacy single-split build_dataset().")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from src.pmut_models.train_utils import train_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for training. Install it first, then run: "
                "pip install -r requirements.txt"
            ) from exc
        raise

    if args.folds:
        # ── 3-fold sweep rotation ──
        folds = build_dataset_folds(
            task=args.task,
            data_root=args.data_root,
            window_size=args.window_size,
            stride=args.stride,
        )
        print(f"Fold protocol: {len(folds)} folds × {args.num_seeds} seeds "
              f"= {len(folds) * args.num_seeds} runs")
        all_fold_accs: list[float] = []
        for fold_idx, bundle in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx}  (train={bundle.metadata.get('train_sweeps',['?'])[0]}, "
                  f"val={bundle.metadata.get('val_sweeps',['?'])[0]}, "
                  f"test={bundle.metadata.get('test_sweeps',['?'])[0]})")
            print(f"  Train: {bundle.x_train.shape[0]}  Val: {bundle.x_val.shape[0]}  "
                  f"Test: {bundle.x_test.shape[0]}")
            print(f"{'='*60}")

            fold_output_dir = (
                args.output_root / "runs"
                / f"{args.task}_folds_w{args.window_size}"
                / args.model / f"fold{fold_idx}"
            )
            metrics = train_model(
                bundle=bundle,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                output_dir=fold_output_dir,
                seed=args.seed,
                num_seeds=args.num_seeds,
                encode=args.encode,
                encode_steps=args.encode_steps,
                encode_channels=args.encode_channels,
            )
            if args.num_seeds > 1:
                acc = metrics["final_test_accuracy_mean"]
                print(f"fold{fold_idx}={acc:.4f}")
                all_fold_accs.append(acc)
            else:
                print(f"fold{fold_idx}={metrics['final_test_accuracy']:.4f}")

        if all_fold_accs:
            print(f"\n{'='*60}")
            print(f"Cross-fold summary ({args.num_seeds} seeds/fold):")
            print(f"  Per-fold:  {[f'{a:.4f}' for a in all_fold_accs]}")
            print(f"  Mean ± std: {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")
            print(f"{'='*60}")
    else:
        # ── Legacy single-split path ──
        bundle = build_dataset(
            task=args.task,
            split=args.split,
            data_root=args.data_root,
            window_size=args.window_size,
            stride=args.stride,
        )
        print(f"Dataset: {len(bundle.labels)} classes, {bundle.x_train.shape[0]} train, {bundle.x_test.shape[0]} test")
        print(f"Labels: {bundle.labels}")
        output_dir = (
            args.output_root
            / "runs"
            / f"{args.task}_{args.split}_{args.model}_w{args.window_size}_seed{args.seed}"
        )
        metrics = train_model(
            bundle=bundle,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=output_dir,
            seed=args.seed,
            num_seeds=args.num_seeds,
            encode=args.encode,
            encode_steps=args.encode_steps,
            encode_channels=args.encode_channels,
        )
        print(f"saved={output_dir}")
        if args.num_seeds > 1:
            print(f"final_test_accuracy={metrics['final_test_accuracy_mean']:.4f} +/- {metrics['final_test_accuracy_std']:.4f}")
            print(f"best_val_accuracy={metrics['best_val_accuracy_mean']:.4f} +/- {metrics['best_val_accuracy_std']:.4f}")
        else:
            print(f"final_test_accuracy={metrics['final_test_accuracy']:.4f}")
            print(f"best_val_accuracy={metrics['best']['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
