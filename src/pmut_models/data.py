import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DATA_ROOT


PURE_MEDIUM_MAP = {
    "AIR": "Air",
    "WATER": "DI Water",
    "100ACETONE": "Acetone",
    "100IPA": "IPA",
    "50IPA50DI": "50% IPA",
    "50DI50ACETONE": "50% Acetone",
}

SIGNAL_COLUMNS = ["amplitude", "phase", "x_v", "y_v"]


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: list[str]
    channels: list[str]
    train_groups: np.ndarray
    test_groups: np.ndarray
    metadata: dict


def _file_paths(data_root: Path) -> dict[str, Path]:
    return {
        "liquid": data_root / "Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv",
        "high": data_root / "pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx",
        "low": data_root / "low_conc_glucose_sweeps_data_2mM_to_50mM.xlsx",
    }


def load_liquid_loading_pure(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_csv(_file_paths(data_root)["liquid"])
    raw = raw[raw["medium"].isin(PURE_MEDIUM_MAP)].copy()
    raw["label"] = raw["medium"].map(PURE_MEDIUM_MAP)
    raw["source"] = "liquid_loading_csv"
    raw["sweep_key"] = raw["medium"].astype(str) + "_" + raw["chunk_timestamp"].astype(str)
    raw["frequency_hz"] = raw["frequency_hz"]
    raw["amplitude"] = raw["amplitude_v"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x_v"]
    raw["y_v"] = raw["y_v"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]


def _normalize_air_di_from_high(condition: str) -> str | None:
    if condition.startswith("Air"):
        return "Air"
    if condition.startswith("DI Water"):
        return "DI Water"
    return None


def load_high_air_di(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_excel(_file_paths(data_root)["high"], sheet_name="AllSamples")
    raw["label"] = raw["condition"].map(_normalize_air_di_from_high)
    raw = raw[raw["label"].notna()].copy()
    raw["source"] = "high_glucose_xlsx"
    raw["sweep_key"] = raw["condition"].astype(str)
    raw["frequency_hz"] = raw["frequency"]
    raw["amplitude"] = raw["r"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x"]
    raw["y_v"] = raw["y"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]


def load_low_air_di(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    path = _file_paths(data_root)["low"]
    frames = []
    for sheet_name in pd.ExcelFile(path).sheet_names:
        lowered = sheet_name.lower()
        if "self" in lowered or "tx" not in lowered or "rx" not in lowered:
            continue
        label = None
        if "air" in lowered:
            label = "Air"
        elif "di_water" in lowered:
            label = "DI Water"
        if label is None:
            continue
        data = pd.read_excel(path, sheet_name=sheet_name)
        data = data.rename(columns={
            "Frequency (Hz)": "frequency_hz",
            "Amplitude": "amplitude",
            "Phase (rad)": "phase_rad",
        })
        data["phase"] = np.rad2deg(data["phase_rad"])
        data["x_v"] = data["amplitude"] * np.cos(data["phase_rad"])
        data["y_v"] = data["amplitude"] * np.sin(data["phase_rad"])
        data["source"] = "low_glucose_xlsx"
        data["sweep_key"] = sheet_name
        data["label"] = label
        frames.append(data[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def _normalize_high_condition(condition: str) -> str | None:
    if condition.startswith("Air"):
        return "Air"
    if condition.startswith("DI Water"):
        return "DI Water"
    match = re.search(r"Glucose\s+(\d+)mM", condition)
    if match:
        return f"Glucose {match.group(1)} mM"
    return None


def load_high_glucose(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_excel(_file_paths(data_root)["high"], sheet_name="AllSamples")
    raw["label"] = raw["condition"].map(_normalize_high_condition)
    raw = raw[raw["label"].notna()].copy()
    raw["source"] = "high_glucose_xlsx"
    raw["sweep_key"] = raw["condition"].astype(str)
    raw["frequency_hz"] = raw["frequency"]
    raw["amplitude"] = raw["r"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x"]
    raw["y_v"] = raw["y"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]


def load_low_glucose(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    path = _file_paths(data_root)["low"]
    xl = pd.ExcelFile(path)
    frames = []
    for sheet_name in xl.sheet_names:
        lowered = sheet_name.lower()
        if "self" in lowered or "tx" not in lowered or "rx" not in lowered:
            continue
        data = pd.read_excel(path, sheet_name=sheet_name)
        if "air" in lowered:
            label = "Air"
        elif "di_water" in lowered:
            label = "DI Water"
        else:
            match = re.search(r"(\d+)\s*mm", lowered, flags=re.IGNORECASE)
            if not match:
                continue
            label = f"Glucose {match.group(1)} mM"
        data = data.rename(
            columns={
                "Frequency (Hz)": "frequency_hz",
                "Amplitude": "amplitude",
                "Phase (rad)": "phase_rad",
            }
        )
        data["phase"] = np.rad2deg(data["phase_rad"])
        data["x_v"] = data["amplitude"] * np.cos(data["phase_rad"])
        data["y_v"] = data["amplitude"] * np.sin(data["phase_rad"])
        data["source"] = "low_glucose_xlsx"
        data["sweep_key"] = sheet_name
        data["label"] = label
        frames.append(data[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def load_task_frame(task: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    if task == "pure":
        liquid = load_liquid_loading_pure(data_root)
        high = load_high_glucose(data_root)
        glucose_1000 = high[high["label"] == "Glucose 1000 mM"].copy()
        high_air_di = load_high_air_di(data_root)
        low_air_di = load_low_air_di(data_root)
        return pd.concat([liquid, glucose_1000, high_air_di, low_air_di], ignore_index=True)
    if task == "high-glucose":
        high = load_high_glucose(data_root)
        return high[high["label"].str.startswith("Glucose")].copy()
    if task == "glucose-ladder":
        high = load_high_glucose(data_root)
        low = load_low_glucose(data_root)
        high = high[high["label"].str.startswith("Glucose")].copy()
        low = low[low["label"].str.startswith("Glucose")].copy()
        return pd.concat([low, high], ignore_index=True)
    raise ValueError(f"Unknown task: {task}")


def make_sequence_samples(frame: pd.DataFrame, window_size: int, stride: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride is None:
        stride = window_size  # non-overlapping windows by default
    sequences = []
    labels = []
    groups = []
    for (_, sweep_key, label), group in frame.groupby(["source", "sweep_key", "label"], sort=False):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        values = group[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - window_size + 1, stride):
            sequences.append(values[start : start + window_size])
            labels.append(label)
            groups.append(sweep_key)
    return np.stack(sequences), np.array(labels), np.array(groups)


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(labels.tolist()), key=_label_sort_key)
    lookup = {label: idx for idx, label in enumerate(classes)}
    return np.array([lookup[label] for label in labels], dtype=np.int64), classes


def _label_sort_key(label: str) -> tuple[int, float | str]:
    match = re.search(r"Glucose\s+(\d+)", label)
    if match:
        return (1, float(match.group(1)))
    return (0, label)


def split_indices(labels: np.ndarray, groups: np.ndarray, split: str, test_every: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if split == "window":
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            mask = np.arange(len(idx)) % test_every == test_every - 1
            if not mask.any():
                mask[-1] = True
            test_idx.extend(idx[mask].tolist())
            train_idx.extend(idx[~mask].tolist())
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    if split == "random_window":
        rng = np.random.RandomState(42)
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            perm = rng.permutation(len(idx))
            n_test = max(1, len(idx) // test_every)
            test_positions = set(perm[:n_test].tolist())
            for j, global_idx in enumerate(idx):
                if j in test_positions:
                    test_idx.append(int(global_idx))
                else:
                    train_idx.append(int(global_idx))
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    if split == "grouped":
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            label_groups = sorted(set(groups[idx].tolist()))
            if len(label_groups) < 2:
                continue
            holdout = label_groups[-1]
            test_mask = groups[idx] == holdout
            test_idx.extend(idx[test_mask].tolist())
            train_idx.extend(idx[~test_mask].tolist())
        if not train_idx:
            raise ValueError("Grouped split needs at least two groups for at least one class.")
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    raise ValueError(f"Unknown split: {split}")


def standardize_from_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def build_dataset(
    task: str,
    split: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    window_size: int = 256,
    stride: int | None = None,
) -> DatasetBundle:
    frame = load_task_frame(task, data_root)
    x, raw_labels, groups = make_sequence_samples(frame, window_size=window_size, stride=stride)
    y, labels = encode_labels(raw_labels)
    train_idx, test_idx = split_indices(raw_labels, groups, split=split)
    x_train, x_test, mean, std = standardize_from_train(x[train_idx], x[test_idx])
    metadata = {
        "task": task,
        "split": split,
        "window_size": window_size,
        "stride": stride or window_size,
        "n_samples": int(len(x)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    return DatasetBundle(
        x_train=x_train.astype(np.float32),
        y_train=y[train_idx],
        x_test=x_test.astype(np.float32),
        y_test=y[test_idx],
        labels=labels,
        channels=SIGNAL_COLUMNS,
        train_groups=groups[train_idx],
        test_groups=groups[test_idx],
        metadata=metadata,
    )


def save_dataset(bundle: DatasetBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        labels=np.array(bundle.labels),
        channels=np.array(bundle.channels),
        train_groups=bundle.train_groups,
        test_groups=bundle.test_groups,
        metadata=json.dumps(bundle.metadata),
    )


def load_npz_dataset(path: Path) -> DatasetBundle:
    data = np.load(path, allow_pickle=False)
    return DatasetBundle(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        labels=data["labels"].tolist(),
        channels=data["channels"].tolist(),
        train_groups=data["train_groups"],
        test_groups=data["test_groups"],
        metadata=json.loads(str(data["metadata"])),
    )

