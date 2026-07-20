import argparse
from pathlib import Path

from src.pmut_models.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from src.pmut_models.data import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PMUT SNN/CNN benchmark models.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--split", choices=["window", "random_window", "grouped"], default="window")
    parser.add_argument(
        "--model",
        choices=["rate_snn_mlp", "temporal_lif_snn", "spiking_cnn1d", "cnn1d"],
        default="rate_snn_mlp",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None, help="Window stride (default: window_size//2 for 50% overlap)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds for mean/std reporting")
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
    )
    print(f"saved={output_dir}")
    if args.num_seeds > 1:
        print(f"final_test_accuracy={metrics['final_test_accuracy_mean']:.4f} +/- {metrics['final_test_accuracy_std']:.4f}")
        print(f"best_test_accuracy={metrics['best_test_accuracy_mean']:.4f} +/- {metrics['best_test_accuracy_std']:.4f}")
    else:
        print(f"final_test_accuracy={metrics['final_test_accuracy']:.4f}")
        print(f"best_test_accuracy={metrics['best']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
