import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import load_task_frame


PLOT_COLORS = {
    "amplitude": "#1f77b4",
    "phase": "#d62728",
    "x_v": "#2ca02c",
    "y_v": "#9467bd",
}


class RawDataViewer(tk.Tk):
    def __init__(self, task: str, data_root: Path):
        super().__init__()
        self.title("PMUT Raw Sweep Viewer")
        self.geometry("1120x760")
        self.minsize(900, 620)

        self.task = tk.StringVar(value=task)
        self.channel = tk.StringVar(value="amplitude")
        self.data_root = data_root
        self.frame = None
        self.sweep_keys: list[str] = []
        self.labels: list[str] = []
        self.selected_label = tk.StringVar(value="")
        self.selected_sweep = tk.StringVar(value="")

        self._build_layout()
        self._load_task()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w")
        task_box = ttk.Combobox(
            controls,
            textvariable=self.task,
            values=["pure", "high-glucose", "glucose-ladder"],
            state="readonly",
            width=18,
        )
        task_box.grid(row=0, column=1, padx=(6, 16), sticky="w")
        task_box.bind("<<ComboboxSelected>>", lambda _event: self._load_task())

        ttk.Label(controls, text="Label").grid(row=0, column=2, sticky="w")
        self.label_box = ttk.Combobox(controls, textvariable=self.selected_label, state="readonly", width=22)
        self.label_box.grid(row=0, column=3, padx=(6, 16), sticky="w")
        self.label_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_sweeps())

        ttk.Label(controls, text="Sweep").grid(row=0, column=4, sticky="w")
        self.sweep_box = ttk.Combobox(controls, textvariable=self.selected_sweep, state="readonly", width=42)
        self.sweep_box.grid(row=0, column=5, padx=(6, 16), sticky="ew")
        self.sweep_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw())

        ttk.Label(controls, text="Channel").grid(row=0, column=6, sticky="w")
        channel_box = ttk.Combobox(
            controls,
            textvariable=self.channel,
            values=["amplitude", "phase", "x_v", "y_v"],
            state="readonly",
            width=12,
        )
        channel_box.grid(row=0, column=7, padx=(6, 0), sticky="w")
        channel_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw())

        controls.columnconfigure(5, weight=1)

        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="white", highlightthickness=1, highlightbackground="#c7c7c7")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        self.summary = tk.Text(body, height=7, wrap="word")
        self.summary.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self.summary.configure(state="disabled")

    def _load_task(self) -> None:
        self.frame = load_task_frame(self.task.get(), self.data_root)
        self.labels = sorted(self.frame["label"].unique().tolist(), key=self._label_sort_key)
        self.label_box.configure(values=self.labels)
        self.selected_label.set(self.labels[0] if self.labels else "")
        self._refresh_sweeps()

    def _refresh_sweeps(self) -> None:
        if self.frame is None:
            return
        label = self.selected_label.get()
        subset = self.frame[self.frame["label"] == label]
        self.sweep_keys = sorted(subset["sweep_key"].unique().tolist())
        self.sweep_box.configure(values=self.sweep_keys)
        self.selected_sweep.set(self.sweep_keys[0] if self.sweep_keys else "")
        self._redraw()

    @staticmethod
    def _label_sort_key(label: str) -> tuple[int, float | str]:
        if label.startswith("Glucose"):
            value = float(label.split()[1])
            return (1, value)
        return (0, label)

    def _current_sweep(self):
        if self.frame is None:
            return None
        label = self.selected_label.get()
        sweep = self.selected_sweep.get()
        data = self.frame[(self.frame["label"] == label) & (self.frame["sweep_key"] == sweep)].copy()
        if data.empty:
            return None
        return data.sort_values("frequency_hz")

    def _redraw(self) -> None:
        data = self._current_sweep()
        self.canvas.delete("all")
        if data is None:
            self._set_summary("No data selected.")
            return

        width = max(self.canvas.winfo_width(), 300)
        height = max(self.canvas.winfo_height(), 260)
        margin_left, margin_right, margin_top, margin_bottom = 78, 28, 38, 64
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        x = data["frequency_hz"].to_numpy(dtype=float)
        y = data[self.channel.get()].to_numpy(dtype=float)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        def sx(value: float) -> float:
            return margin_left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return margin_top + (y_max - value) / (y_max - y_min) * plot_h

        self._draw_axes(width, height, margin_left, margin_right, margin_top, margin_bottom, x_min, x_max, y_min, y_max)

        coords = []
        step = max(1, len(x) // 2200)
        for xv, yv in zip(x[::step], y[::step]):
            coords.extend([sx(float(xv)), sy(float(yv))])
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=PLOT_COLORS[self.channel.get()], width=2)

        title = f"{self.task.get()} | {self.selected_label.get()} | {self.selected_sweep.get()} | {self.channel.get()}"
        self.canvas.create_text(width // 2, 18, text=title, font=("Segoe UI", 11, "bold"), fill="#222222")

        summary = (
            f"Task: {self.task.get()}\n"
            f"Label: {self.selected_label.get()}\n"
            f"Sweep: {self.selected_sweep.get()}\n"
            f"Points: {len(data):,}\n"
            f"Frequency: {x_min:,.0f} Hz to {x_max:,.0f} Hz\n"
            f"{self.channel.get()}: min={y_min:.6g}, max={y_max:.6g}, mean={float(np.mean(y)):.6g}, std={float(np.std(y)):.6g}"
        )
        self._set_summary(summary)

    def _draw_axes(
        self,
        width: int,
        height: int,
        margin_left: int,
        margin_right: int,
        margin_top: int,
        margin_bottom: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        left = margin_left
        right = width - margin_right
        top = margin_top
        bottom = height - margin_bottom
        self.canvas.create_rectangle(left, top, right, bottom, outline="#333333")

        for i in range(6):
            frac = i / 5
            x_pos = left + frac * (right - left)
            value = x_min + frac * (x_max - x_min)
            self.canvas.create_line(x_pos, bottom, x_pos, bottom + 5, fill="#333333")
            self.canvas.create_text(x_pos, bottom + 20, text=f"{value/1e6:.2f}M", font=("Segoe UI", 9), fill="#333333")

            y_pos = bottom - frac * (bottom - top)
            y_value = y_min + frac * (y_max - y_min)
            self.canvas.create_line(left - 5, y_pos, left, y_pos, fill="#333333")
            self.canvas.create_text(left - 10, y_pos, text=f"{y_value:.3g}", anchor="e", font=("Segoe UI", 9), fill="#333333")

        self.canvas.create_text((left + right) / 2, height - 22, text="Frequency (Hz)", font=("Segoe UI", 10), fill="#222222")
        self.canvas.create_text(18, (top + bottom) / 2, text=self.channel.get(), angle=90, font=("Segoe UI", 10), fill="#222222")

    def _set_summary(self, text: str) -> None:
        self.summary.configure(state="normal")
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a Tkinter interface for PMUT raw sweep visualization.")
    parser.add_argument("--task", choices=["high-glucose", "glucose-ladder"], default="high-glucose")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = RawDataViewer(task=args.task, data_root=args.data_root)
    app.mainloop()


if __name__ == "__main__":
    main()

