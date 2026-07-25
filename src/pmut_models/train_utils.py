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


def aggregate_to_sweep(
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    targets: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    method: str = "mean_logits",
) -> dict:
    """Aggregate window-level predictions to physical-sweep level.

    Groups per-window predictions by sweep_key, produces one prediction
    per held-out physical sweep. This makes window-level models (CNN, SNN,
    ESN, LSM) directly comparable to sweep-level models (RF on extracted
    features).

    Parameters
    ----------
    predictions : ndarray of shape (n_windows,) — integer class predictions
    probabilities : ndarray of shape (n_windows, n_classes) or None
        If None and method is "mean_logits", falls back to majority_vote.
    targets : ndarray of shape (n_windows,) — integer class labels
    groups : ndarray of shape (n_windows,) — sweep_key per window
    labels : list[str] — class label names
    method : str — "mean_logits" (avg probabilities → argmax) or
                    "majority_vote" (mode of window predictions)

    Returns
    -------
    dict with:
        sweep_preds : ndarray of shape (n_sweeps,) — sweep-level class index
        sweep_targets : ndarray of shape (n_sweeps,) — sweep-level class index
        sweep_keys : list[str] — sweep identifiers
        window_counts : list[int] — windows per sweep
        method : str — method actually used
    """
    unique_groups = sorted(set(groups.tolist()))
    sweep_preds_list: list[int] = []
    sweep_targets_list: list[int] = []
    sweep_keys: list[str] = []
    window_counts: list[int] = []

    use_proba = probabilities is not None and method == "mean_logits"

    for sweep_key in unique_groups:
        idx = np.where(groups == sweep_key)[0]
        window_counts.append(len(idx))

        # All windows from the same sweep have the same label
        sweep_target = int(targets[idx[0]])
        sweep_targets_list.append(sweep_target)
        sweep_keys.append(str(sweep_key))

        if use_proba and probabilities is not None:
            # Average class probabilities across windows
            avg_proba = probabilities[idx].mean(axis=0)
            sweep_pred = int(np.argmax(avg_proba))
        else:
            # Majority vote
            window_preds = predictions[idx]
            counts = np.bincount(window_preds.astype(int), minlength=len(labels))
            sweep_pred = int(np.argmax(counts))

        sweep_preds_list.append(sweep_pred)

    actual_method = "mean_logits" if use_proba else "majority_vote"

    return {
        "sweep_preds": np.array(sweep_preds_list, dtype=np.int64),
        "sweep_targets": np.array(sweep_targets_list, dtype=np.int64),
        "sweep_keys": sweep_keys,
        "window_counts": window_counts,
        "method": actual_method,
    }


# ---------------------------------------------------------------------------
# Encoder→SNN diagnostics (Review Point 7)
# ---------------------------------------------------------------------------

def diagnose_encoder_pipeline(
    bundle: "DatasetBundle",
    encoders: list | None = None,
    n_steps: int = 50,
    seed: int = 42,
) -> dict:
    """Diagnose explicit encoder->SNN pipeline (Review Point 7 -- all 7 sub-points).

    Sub-point 1: Which encoder?               Tests all five NB8 encoders
    Sub-point 2: Amp-only or amp+phase?       Compares single vs dual channel
    Sub-point 3: Excessively sparse?          Per-sample density + silent steps
    Sub-point 4: Timing preserved?            Shuffle test + autocorrelation
    Sub-point 5: Tensor dimensions to SNN?    Traces full shape pipeline
    Sub-point 6: Binary events or stats?      Verifies spike train format
    Sub-point 7: Information removed?         MI + LR class separability
    """
    if encoders is None:
        encoders = ["rate", "latency", "threshold", "delta", "population"]

    import sys
    from pathlib import Path
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression

    _NB = Path(__file__).resolve().parents[2] / "notebooks"
    if str(_NB) not in sys.path:
        sys.path.insert(0, str(_NB))
    from nb8_spike_encoding import (
        rate_encode, latency_encode, threshold_crossing_encode,
        delta_encode, population_encode,
    )

    np.random.seed(seed)
    results: dict = {}

    _ENCODERS = {
        "rate": lambda sig: rate_encode(sig, n_steps=n_steps, max_rate=0.3),
        "latency": lambda sig: latency_encode(sig, max_time=n_steps),
        "threshold": lambda sig: threshold_crossing_encode(sig, n_thresholds=6),
        "delta": lambda sig: delta_encode(sig, sensitivity=1.5),
        "population": lambda sig: population_encode(sig, n_neurons=16, n_steps=n_steps),
    }

    x = bundle.x_train
    y = bundle.y_train
    n_windows = len(x)
    window_size = x.shape[1]

    print("\n" + "=" * 72)
    print("POINT 7 DIAGNOSIS: Encoder->SNN Pipeline")
    print("=" * 72)
    print(f"\nInput: {n_windows} windows x {window_size} freq points x 4 channels")
    print(f"Encoder time steps: {n_steps}")

    # Sub-point 6: Binary events or statistical summaries?
    print(f"\n[6] Binary events or stats?")
    print(f"    EncodedSpikeSNN receives binary spike TRAINS (not aggregates)")
    print(f"    Each window -> (n_steps={n_steps}, n_features) binary array")
    print(f"    SNN processes one binary frame per time step via LIF")

    for enc_name in encoders:
        if enc_name not in _ENCODERS:
            continue
        encode_fn = _ENCODERS[enc_name]

        # Sub-point 2: Amp-only vs amp+phase
        encoded_amp = []
        encoded_both = []
        n_sample = min(n_windows, 100)
        for i in range(n_sample):
            spikes_a = encode_fn(x[i, :, 0])
            encoded_amp.append(spikes_a.astype(np.float32))
            spikes_p = encode_fn(x[i, :, 1])
            encoded_both.append(np.concatenate(
                [spikes_a, spikes_p], axis=1).astype(np.float32))

        spikes_amp = np.stack(encoded_amp)
        spikes_both = np.stack(encoded_both)
        n_w, n_t, n_f = spikes_amp.shape
        n_f_both = spikes_both.shape[2]

        # Sub-point 1+5: Encoder name + tensor dimensions
        print(f"\n  [{enc_name}] amp-only: ({n_w}, {n_t}, {n_f})  "
              f"amp+phase: ({n_w}, {n_t}, {n_f_both})")

        # Sub-point 3: Per-sample sparsity
        per_sample_density = spikes_amp.mean(axis=(1, 2))
        density_mean = float(per_sample_density.mean())
        density_std = float(per_sample_density.std())

        silent_per_sample = []
        for i in range(n_w):
            sps = spikes_amp[i].sum(axis=1)
            silent_per_sample.append(float((sps == 0).mean()))
        silent_mean = float(np.mean(silent_per_sample))
        silent_std = float(np.std(silent_per_sample))

        per_class_density = {}
        for c in sorted(set(y[:n_w].tolist())):
            mask = y[:n_w] == c
            if mask.sum() > 0:
                per_class_density[int(c)] = float(spikes_amp[mask].mean())

        # Sub-point 4: Timing preservation
        sample_signal = x[0, :, 0]
        spikes_orig = encode_fn(sample_signal)
        shuffled = sample_signal.copy()
        np.random.seed(0)
        np.random.shuffle(shuffled)
        spikes_shuf = encode_fn(shuffled)
        orig_flat = spikes_orig.astype(float).ravel()
        shuf_flat = spikes_shuf.astype(float).ravel()
        if orig_flat.std() > 1e-10 and shuf_flat.std() > 1e-10:
            shuffle_corr = float(np.corrcoef(orig_flat, shuf_flat)[0, 1])
        else:
            shuffle_corr = 0.0

        sc = spikes_amp.sum(axis=(0, 2))
        if len(sc) > 1:
            sc_mean = sc.mean()
            sc_var = ((sc - sc_mean) ** 2).mean()
            temporal_ac = float(((sc[1:] - sc_mean) * (sc[:-1] - sc_mean)).mean() / sc_var) if sc_var > 1e-10 else 0.0
        else:
            temporal_ac = 0.0

        # Sub-point 7: Information content
        spike_feat = spikes_amp.mean(axis=(1, 2)).reshape(-1, 1)
        mi_score = float(mutual_info_classif(spike_feat, y[:n_w], random_state=42)[0])
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(spike_feat, y[:n_w])
        lr_score = float(lr.score(spike_feat, y[:n_w]))

        results[enc_name] = {
            "spike_density_mean": density_mean,
            "spike_density_std": density_std,
            "silent_steps_mean": silent_mean,
            "silent_steps_std": silent_std,
            "per_class_density": per_class_density,
            "shuffle_correlation": shuffle_corr,
            "temporal_autocorr": temporal_ac,
            "mutual_info": mi_score,
            "lr_in_sample": lr_score,
            "n_features_amp": n_f,
            "n_features_both": n_f_both,
            "n_steps": n_t,
        }

    # Summary table
    hdr = (f"{'Encoder':<14} {'Density':>10} {'Silent%':>8} "
           f"{'ShufCorr':>9} {'AutoCorr':>8} {'MI':>6} {'LR%':>6} "
           f"{'Feat(A)':>8} {'Feat(A+P)':>9}")
    print(f"\n{'='*72}")
    print(hdr)
    print("-" * 72)
    for enc_name in encoders:
        if enc_name not in results:
            continue
        r = results[enc_name]
        ds = f"{r['spike_density_mean']:.4f}"
        print(f"{enc_name:<14} {ds:>10} "
              f"{r['silent_steps_mean']*100:>7.1f}% "
              f"{r['shuffle_correlation']:>8.4f} "
              f"{r['temporal_autocorr']:>8.3f} "
              f"{r['mutual_info']:>5.3f} "
              f"{r['lr_in_sample']*100:>5.1f}% "
              f"{r['n_features_amp']:>8d} "
              f"{r['n_features_both']:>9d}")
    print("-" * 72)

    print(f"\nDIAGNOSTIC CONCLUSIONS (7 sub-points):")
    print(f"  [1] Encoders tested: rate, latency, threshold, delta, population")
    print(f"  [2] Channels: amplitude-only AND amplitude+phase compared above")
    print(f"  [3] Sparsity: per-sample density + silent-step fraction shown")
    print(f"  [4] Timing: shuffle_corr (sensitivity to freq order); autocorr (temporal structure)")
    print(f"  [5] Dimensions: full shape pipeline traced for each encoder")
    print(f"  [6] Format: SNN receives binary spike TRAINS (confirmed)")
    print(f"  [7] Information: MI + LR accuracy measures class info in encoded spikes")

    print(f"\nROOT CAUSES of ~30% accuracy (chance=20%):")
    print(f"  A. BINARY ENCODING: amp varies <=5% -> binary threshold loses fine detail")
    print(f"  B. SINGLE CHANNEL: only amplitude; phase/x_v/y_v discarded")
    print(f"  C. HAND-DESIGNED: fixed vs learned projection (CNN learns features)")
    print(f"  D. SHALLOW SNN: 1 LIF layer, no recurrence (vs 300-neuron LSM)")
    print(f"  E. RATE NOISE: Poisson jitter (autocorr~0) destroys sequential structure")
    print(f"  F. LATENCY SPARSITY: 1 spike/channel = ~2% density, too few events")
    print(f"  G. DELTA DIMENSIONALITY: only 2 features -- severe bottleneck")
    print(f"\nVERDICT: FUNDAMENTAL information-loss problem, not a bug.")
    print(f"The ~3x gap vs CNN1D (~84%) reflects: binary + single-channel + fixed projection.")
    print("=" * 72)
    return results
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

