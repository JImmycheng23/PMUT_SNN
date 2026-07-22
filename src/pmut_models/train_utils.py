import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle
from .spiking import build_model, SpikeActivationHook


DEFAULT_SEEDS = [7, 42, 123, 999, 2024]

# ── Named encoders for the --encode flag ──
_ENCODER_REGISTRY: dict[str, str] = {
    "rate": "rate_encode",
    "latency": "latency_encode",
    "threshold": "threshold_crossing_encode",
    "delta": "delta_encode",
    "population": "population_encode",
}


def encode_dataset_bundle(
    bundle: DatasetBundle,
    encoder_name: str,
    n_steps: int = 50,
    seed: int = 42,
    encode_channels: str = "amplitude",
) -> DatasetBundle:
    """Pre-encode every window in a DatasetBundle with an NB8 spike encoder.

    Each selected signal channel is independently encoded, then spike trains
    are concatenated along the feature axis.  The resulting bundle has shape
    ``(n_windows, n_steps, n_features)`` where ``n_features`` depends on the
    encoder and the number of channels.

    Parameters
    ----------
    encode_channels:
        ``"amplitude"`` — channel 0 only.
        ``"phase"``     — channel 1 only.
        ``"amplitude+phase"`` — channels 0 and 1, concatenated.

    Deterministic encoders (latency, threshold, delta) are encoded once.
    Stochastic encoders (rate, population) use a fixed seed for reproducibility.
    """
    _CHANNEL_MAP = {
        "amplitude": [0],
        "phase": [1],
        "amplitude+phase": [0, 1],
    }
    if encode_channels not in _CHANNEL_MAP:
        raise ValueError(
            f"Unknown encode_channels '{encode_channels}'. "
            f"Choices: {list(_CHANNEL_MAP)}"
        )
    channel_ids = _CHANNEL_MAP[encode_channels]

    # Late import so the notebooks directory does not need to be on sys.path
    # at module load time.
    _NB = Path(__file__).resolve().parents[2] / "notebooks"
    if str(_NB) not in sys.path:
        sys.path.insert(0, str(_NB))
    from nb8_spike_encoding import (  # type: ignore[import-not-found]
        rate_encode,
        latency_encode,
        threshold_crossing_encode,
        delta_encode,
        population_encode,
    )

    np.random.seed(seed)

    _ENCODERS = {
        "rate": lambda sig: rate_encode(sig, n_steps=n_steps, max_rate=0.3),
        "latency": lambda sig: latency_encode(sig, max_time=n_steps),
        "threshold": lambda sig: threshold_crossing_encode(sig, n_thresholds=6),
        "delta": lambda sig: delta_encode(sig, sensitivity=1.5),
        "population": lambda sig: population_encode(sig, n_neurons=16, n_steps=n_steps),
    }

    if encoder_name not in _ENCODERS:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Choices: {list(_ENCODERS)}")

    encode_fn = _ENCODERS[encoder_name]

    def _encode(x: np.ndarray) -> np.ndarray:
        """x: (n_windows, window_size, 4) → (n_windows, n_steps, total_features)"""
        encoded = []
        for i in range(len(x)):
            channel_spikes = []
            for ch_id in channel_ids:
                signal = x[i, :, ch_id]
                spikes = encode_fn(signal)
                channel_spikes.append(spikes.astype(np.float32))
            # Concatenate along feature axis (axis=1)
            combined = np.concatenate(channel_spikes, axis=1)
            encoded.append(combined)
        return np.stack(encoded)

    x_train_enc = _encode(bundle.x_train)
    x_test_enc = _encode(bundle.x_test)

    # Encode validation data if present
    x_val_enc = None
    y_val_enc = None
    val_groups_enc = None
    if bundle.x_val is not None and bundle.y_val is not None:
        x_val_enc = _encode(bundle.x_val)
        y_val_enc = bundle.y_val
        val_groups_enc = bundle.val_groups

    return DatasetBundle(
        x_train=x_train_enc,
        y_train=bundle.y_train,
        x_test=x_test_enc,
        y_test=bundle.y_test,
        x_val=x_val_enc,
        y_val=y_val_enc,
        val_groups=val_groups_enc,
        labels=bundle.labels,
        channels=bundle.channels,
        train_groups=bundle.train_groups,
        test_groups=bundle.test_groups,
        metadata={
            **bundle.metadata,
            "encoder": encoder_name,
            "n_steps": n_steps,
            "encode_seed": seed,
            "encode_channels": encode_channels,
        },
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(bundle: DatasetBundle, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ds = TensorDataset(torch.from_numpy(bundle.x_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.x_test), torch.from_numpy(bundle.y_test))
    val_loader = None
    if bundle.x_val is not None and bundle.y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(bundle.x_val), torch.from_numpy(bundle.y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        val_loader,
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
    train_loader, test_loader, val_loader = make_loaders(bundle, batch_size=batch_size)
    input_shape = (bundle.x_train.shape[1], bundle.x_train.shape[2])

    # Use validation set if available, otherwise fall back to train set size
    n_val = len(bundle.y_val) if bundle.y_val is not None else 0
    has_val = val_loader is not None

    # encoded_spike_snn needs to know the encoder output dimensionality
    n_features: int | None = None
    if model_name == "encoded_spike_snn":
        n_features = bundle.x_train.shape[2]  # encoder output channels

    model = build_model(
        model_name,
        input_shape=input_shape,
        num_classes=len(bundle.labels),
        n_features=n_features,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = {"epoch": 0, "val_accuracy": -1.0}
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

        # ── Validate: use val set if available, else (legacy) use test set ──
        if has_val:
            val_acc, _, _, _ = evaluate(model, val_loader, device)
        else:
            val_acc, _, _, _ = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "train_accuracy": train_acc / max(n_batches, 1),
            "val_accuracy": val_acc,
        }
        history.append(row)

        # Model selection on VALIDATION accuracy (not test)
        if val_acc > best["val_accuracy"]:
            best = row.copy()
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        eval_label = "val" if has_val else "test"
        print(
            f"seed={seed:04d} epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} {eval_label}_acc={val_acc:.4f}"
        )

    # ── Final evaluation: load best-val checkpoint, evaluate on TEST once ──
    if (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    test_acc, predictions, targets, spike_maps = evaluate(
        model, test_loader, device, collect_spike_maps=True,
    )

    # ── Per-class + aggregate metrics ──
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    bal_acc = float(balanced_accuracy_score(targets, predictions))
    cm = confusion_matrix(targets, predictions)
    report = classification_report(
        targets, predictions, target_names=bundle.labels, output_dict=True, zero_division=0,
    )
    per_class_recall = {
        lbl: float(report[lbl]["recall"]) for lbl in bundle.labels if lbl in report
    }
    macro = report.get("macro avg", {})
    macro_precision = float(macro.get("precision", 0))
    macro_recall = float(macro.get("recall", 0))
    macro_f1 = float(macro.get("f1-score", 0))

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
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "n_val": n_val,
        "has_val": has_val,
        "best": best,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        hdr = ["epoch", "train_loss", "train_accuracy", "val_accuracy"]
        writer = csv.DictWriter(f, fieldnames=hdr)
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
    encode: str | None = None,
    encode_steps: int = 50,
    encode_channels: str = "amplitude",
) -> dict:
    if encode is not None:
        if model_name != "encoded_spike_snn":
            raise ValueError(
                f"--encode {encode} requires --model encoded_spike_snn, got {model_name}"
            )
        bundle = encode_dataset_bundle(
            bundle, encode, n_steps=encode_steps, seed=seed,
            encode_channels=encode_channels,
        )

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
    best_val_accs = [m["best"]["val_accuracy"] for m in all_metrics]
    summary = {
        "model": model_name,
        "num_seeds": num_seeds,
        "seeds": seeds,
        "labels": bundle.labels,
        "dataset": bundle.metadata,
        "final_test_accuracy_mean": float(np.mean(final_accs)),
        "final_test_accuracy_std": float(np.std(final_accs)),
        "final_test_accuracy_list": final_accs,
        "best_val_accuracy_mean": float(np.mean(best_val_accs)),
        "best_val_accuracy_std": float(np.std(best_val_accs)),
        "best_val_accuracy_list": best_val_accs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Multi-seed summary (n={num_seeds}) ===")
    print(f"  Final test accuracy:  {summary['final_test_accuracy_mean']:.4f} +/- {summary['final_test_accuracy_std']:.4f}")
    print(f"  Best val accuracy:    {summary['best_val_accuracy_mean']:.4f} +/- {summary['best_val_accuracy_std']:.4f}")
    print(f"  Individual final:     {[f'{a:.4f}' for a in final_accs]}")
    return summary

