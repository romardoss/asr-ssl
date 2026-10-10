"""Plot Lightning metrics for one fine-tune version.

Reads ``fine-tune/output/<model>/version_<N>/metrics.csv`` and writes PNG
files under ``plots/`` in that same version folder. With ``--watch``, the
folder is refreshed whenever the CSV changes. This script only reads the
metrics file; it does not start or modify training.

    python fine-tune/plot_metrics.py --version 2
    python fine-tune/plot_metrics.py --version 3 --watch
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FINE_TUNE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = FINE_TUNE_DIR / "output"
DEFAULT_MODEL_NAME = "model-whisper-uk"
PLOTS_DIRNAME = "plots"

plt.rcParams.update(
    {
        "figure.figsize": (11, 6.2),
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Directory name under fine-tune/output. Default: model-whisper-uk.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version number or folder name, e.g. 3 or version_3. Default: newest version.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep refreshing plots while metrics.csv changes.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=20.0,
        help="Seconds between checks in --watch mode.",
    )
    return parser.parse_args()


def version_directories(model_dir: Path) -> list[Path]:
    found = []
    for path in model_dir.glob("version_*"):
        suffix = path.name.removeprefix("version_")
        if path.is_dir() and suffix.isdigit():
            found.append(path)
    return sorted(found, key=lambda path: int(path.name.removeprefix("version_")))


def resolve_version_dir(model_name: str, version: str | None) -> Path:
    model_dir = OUTPUT_DIR / model_name
    if not model_dir.is_dir():
        raise SystemExit(f"Model output directory not found: {model_dir}")
    versions = version_directories(model_dir)
    if version is None:
        if not versions:
            raise SystemExit(f"No version_* folders in {model_dir}")
        return versions[-1]
    name = version if version.startswith("version_") else f"version_{version}"
    path = model_dir / name
    if not path.is_dir():
        known = ", ".join(item.name for item in versions) or "(none)"
        raise SystemExit(f"{path} does not exist. Known versions: {known}")
    return path


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def points(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns or "step" not in frame.columns:
        return pd.DataFrame(columns=["step", column])
    selected = frame.loc[frame[column].notna(), ["step", column]].dropna()
    if selected.empty:
        return selected
    return selected.drop_duplicates("step", keep="last").sort_values("step")


def learning_rate_column(frame: pd.DataFrame) -> str | None:
    matches = [column for column in frame.columns if column == "lr" or column.startswith("lr-")]
    return matches[0] if matches else None


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def style_axis(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def set_step_limits(axis: plt.Axes, steps: pd.Series) -> None:
    low = float(steps.min())
    high = float(steps.max())
    if high == low:
        axis.set_xlim(low - 1, high + 1)
        return
    pad = (high - low) * 0.04
    axis.set_xlim(low - pad, high + pad)


def plot_line(
    series: pd.DataFrame,
    column: str,
    path: Path,
    title: str,
    ylabel: str,
    *,
    scale: float = 1.0,
    color: str = "#1f4e79",
) -> bool:
    if series.empty:
        return False
    figure, axis = plt.subplots()
    values = series[column] * scale
    axis.plot(series["step"], values, color=color, marker="o", markersize=4.5)
    last_step = series["step"].iloc[-1]
    last_value = values.iloc[-1]
    axis.annotate(
        f"{last_value:.4g}",
        xy=(last_step, last_value),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        color=color,
    )
    style_axis(axis, title, "step", ylabel)
    set_step_limits(axis, series["step"])
    save_figure(figure, path)
    return True


def plot_error_rates(frame: pd.DataFrame, path: Path) -> bool:
    series = {
        "WER": points(frame, "val_wer"),
        "CER": points(frame, "val_cer"),
        "MER": points(frame, "val_mer"),
    }
    series = {name: values for name, values in series.items() if not values.empty}
    if not series:
        return False
    colors = {"WER": "#9b2226", "CER": "#1d4e89", "MER": "#2a9d8f"}
    figure, axis = plt.subplots()
    steps = []
    for name, values in series.items():
        percents = values.iloc[:, 1] * 100
        axis.plot(
            values["step"],
            percents,
            label=f"{name}  {percents.iloc[-1]:.2f}%",
            color=colors[name],
            marker="o",
            markersize=5,
        )
        steps.append(values["step"])
    style_axis(axis, "Dev set error rates", "step", "error rate (%)")
    set_step_limits(axis, pd.concat(steps))
    axis.legend(frameon=True, title="latest")
    save_figure(figure, path)
    return True


def plot_edit_counts(frame: pd.DataFrame, path: Path) -> bool:
    columns = {
        "substitutions": "val_substitutions",
        "deletions": "val_deletions",
        "insertions": "val_insertions",
    }
    series = {label: points(frame, column) for label, column in columns.items()}
    series = {label: values for label, values in series.items() if not values.empty}
    if not series:
        return False
    colors = {"substitutions": "#9b2226", "deletions": "#e09f3e", "insertions": "#1d4e89"}
    figure, axis = plt.subplots()
    steps = []
    for label, values in series.items():
        counts = values.iloc[:, 1]
        axis.plot(
            values["step"],
            counts,
            label=f"{label}  {counts.iloc[-1]:,.0f}",
            color=colors[label],
            marker="o",
            markersize=5,
        )
        steps.append(values["step"])
    style_axis(axis, "Dev set word edits", "step", "count")
    set_step_limits(axis, pd.concat(steps))
    axis.legend(frameon=True, title="latest")
    save_figure(figure, path)
    return True


def plot_test_rates(frame: pd.DataFrame, path: Path) -> bool:
    labels = []
    values = []
    for label, column in (("WER", "test_wer"), ("CER", "test_cer"), ("MER", "test_mer")):
        if column not in frame.columns or frame[column].dropna().empty:
            continue
        labels.append(label)
        values.append(float(frame[column].dropna().iloc[-1]) * 100)
    if not labels:
        return False
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(labels, values, color=["#9b2226", "#1d4e89", "#2a9d8f"][: len(labels)], width=0.6)
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )
    style_axis(axis, "Test set error rates", "", "error rate (%)")
    axis.set_ylim(0, max(values) * 1.25 if values else 1)
    save_figure(figure, path)
    return True


def write_plots(version_dir: Path) -> list[Path]:
    metrics_path = version_dir / "metrics.csv"
    plots_dir = version_dir / PLOTS_DIRNAME
    frame = load_metrics(metrics_path)
    written: list[Path] = []

    def keep(created: bool, name: str) -> None:
        path = plots_dir / name
        if created:
            written.append(path)
        elif path.is_file():
            path.unlink()

    keep(
        plot_line(
            points(frame, "train_loss_step"),
            "train_loss_step",
            plots_dir / "train_loss.png",
            "Training loss",
            "loss",
        ),
        "train_loss.png",
    )
    rate_column = learning_rate_column(frame)
    keep(
        plot_line(
            points(frame, rate_column) if rate_column else pd.DataFrame(),
            rate_column or "lr",
            plots_dir / "learning_rate.png",
            "Learning rate",
            "learning rate",
            color="#6a4c93",
        ),
        "learning_rate.png",
    )
    keep(
        plot_line(
            points(frame, "val_loss"),
            "val_loss",
            plots_dir / "val_loss.png",
            "Dev set loss",
            "loss",
            color="#bc4749",
        ),
        "val_loss.png",
    )
    keep(plot_error_rates(frame, plots_dir / "val_error_rates.png"), "val_error_rates.png")
    keep(plot_edit_counts(frame, plots_dir / "val_edit_counts.png"), "val_edit_counts.png")
    keep(plot_test_rates(frame, plots_dir / "test_error_rates.png"), "test_error_rates.png")
    epoch_loss = points(frame, "train_loss_epoch")
    keep(
        plot_line(
            epoch_loss,
            "train_loss_epoch",
            plots_dir / "train_loss_epoch.png",
            "Training loss by epoch",
            "loss",
            color="#386641",
        ),
        "train_loss_epoch.png",
    )
    return written


def fingerprint(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def training_is_running() -> bool:
    # A separate watcher must not signal the trainer. Existence of the process
    # only decides when this watcher itself can stop.
    needle = "fine-tune/fine-tune.py"
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = entry.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if needle in command and "plot_metrics.py" not in command:
            return True
    return False


def watch(version_dir: Path, interval: float) -> None:
    metrics_path = version_dir / "metrics.csv"
    seen: tuple[int, int] | None = None
    print(f"watching {metrics_path}", flush=True)
    while True:
        current = fingerprint(metrics_path)
        if current is not None and current != seen:
            written = write_plots(version_dir)
            seen = current
            print(f"wrote {len(written)} plot(s) to {version_dir / PLOTS_DIRNAME}", flush=True)
        elif current is None:
            print(f"waiting for {metrics_path.name}", flush=True)
        if not training_is_running() and current == seen:
            print("training is not running; plot watcher stopped", flush=True)
            return
        time.sleep(interval)


def main() -> None:
    args = parse_args()
    version_dir = resolve_version_dir(args.model_name, args.version)
    if args.watch:
        watch(version_dir, args.interval)
        return
    metrics_path = version_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise SystemExit(f"Metrics file not found: {metrics_path}")
    written = write_plots(version_dir)
    if not written:
        raise SystemExit(f"No plottable rows in {metrics_path}")
    for path in written:
        print(path)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
