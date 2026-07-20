import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle
from .spiking import build_model, SpikeActivationHook


DEFAULT_SEEDS = [7, 42, 123, 999, 2024]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(bundle: DatasetBundle, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(bundle.x_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.x_test), torch.from_numpy(bundle.y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, collect_spike_maps: bool = False) -> tuple[float, np.ndarray, np.ndarray, dict | None]:
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets_all = []
    spike_maps = None
    if collect_spike_maps and hasattr(model, "register_spike_hooks"):
        hooks = model.register_spike_hooks()
    else:
        hooks = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            predictions.extend(pred.cpu().numpy().tolist())
            targets_all.extend(y.cpu().numpy().tolist())
    if hooks:
        spike_maps = {hook.name: hook.get_activation_map().numpy() for hook in hooks}
    return correct / max(total, 1), np.array(predictions), np.array(targets_all), spike_maps


def train_single_run(
    bundle: DatasetBundle,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    seed: int = 7,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(bundle, batch_size=batch_size)
    input_shape = (bundle.x_train.shape[1], bundle.x_train.shape[2])
    model = build_model(model_name, input_shape=input_shape, num_classes=len(bundle.labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = {"epoch": 0, "test_accuracy": -1.0}
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_acc += accuracy_from_logits(logits.detach(), y)
            n_batches += 1
        test_acc, _, _, _ = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "train_accuracy": train_acc / max(n_batches, 1),
            "test_accuracy": test_acc,
        }
        history.append(row)
        if test_acc > best["test_accuracy"]:
            best = row.copy()
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        print(
            f"seed={seed:04d} epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}"
        )

    test_acc, predictions, targets, spike_maps = evaluate(model, test_loader, device, collect_spike_maps=True)
    metrics = {
        "seed": seed,
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "labels": bundle.labels,
        "dataset": bundle.metadata,
        "final_test_accuracy": test_acc,
        "best": best,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "test_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "prediction", "target_label", "prediction_label"])
        writer.writeheader()
        for target, pred in zip(targets.tolist(), predictions.tolist()):
            writer.writerow(
                {
                    "target": target,
                    "prediction": pred,
                    "target_label": bundle.labels[target],
                    "prediction_label": bundle.labels[pred],
                }
            )
    if spike_maps:
        np.savez_compressed(output_dir / "spike_maps.npz", **spike_maps)
    return metrics


def train_model(
    bundle: DatasetBundle,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    seed: int = 7,
    num_seeds: int = 1,
) -> dict:
    if num_seeds <= 1:
        return train_single_run(
            bundle=bundle,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=output_dir,
            seed=seed,
        )

    seeds = DEFAULT_SEEDS[:num_seeds] if seed == 7 else [seed + i for i in range(num_seeds)]
    all_metrics = []
    for s in seeds:
        run_dir = output_dir.parent / f"{output_dir.name}_seed{s}"
        m = train_single_run(
            bundle=bundle,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=run_dir,
            seed=s,
        )
        all_metrics.append(m)

    final_accs = [m["final_test_accuracy"] for m in all_metrics]
    best_accs = [m["best"]["test_accuracy"] for m in all_metrics]
    summary = {
        "model": model_name,
        "num_seeds": num_seeds,
        "seeds": seeds,
        "labels": bundle.labels,
        "dataset": bundle.metadata,
        "final_test_accuracy_mean": float(np.mean(final_accs)),
        "final_test_accuracy_std": float(np.std(final_accs)),
        "final_test_accuracy_list": final_accs,
        "best_test_accuracy_mean": float(np.mean(best_accs)),
        "best_test_accuracy_std": float(np.std(best_accs)),
        "best_test_accuracy_list": best_accs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Multi-seed summary (n={num_seeds}) ===")
    print(f"  Final test accuracy: {summary['final_test_accuracy_mean']:.4f} +/- {summary['final_test_accuracy_std']:.4f}")
    print(f"  Best test accuracy:  {summary['best_test_accuracy_mean']:.4f} +/- {summary['best_test_accuracy_std']:.4f}")
    print(f"  Individual final:    {[f'{a:.4f}' for a in final_accs]}")
    return summary

