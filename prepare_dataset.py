import argparse
from pathlib import Path

from src.pmut_models.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from src.pmut_models.data import build_dataset, save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PMUT sweep datasets as NPZ arrays.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--split", choices=["window", "random_window", "grouped"], default="window")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None, help="Window stride (default: window_size//2 for 50% overlap)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_dataset(
        task=args.task,
        split=args.split,
        data_root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
    )
    output_path = args.output_root / "datasets" / f"{args.task}_{args.split}_w{args.window_size}.npz"
    save_dataset(bundle, output_path)
    print(f"saved={output_path}")
    print(f"labels={bundle.labels}")
    print(f"x_train={bundle.x_train.shape} x_test={bundle.x_test.shape}")


if __name__ == "__main__":
    main()

