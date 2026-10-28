"""Compare word and character error rate across every evaluation in a folder.

Each subdirectory with a ``results.json`` is one model. Datasets sit on the
x-axis. One panel is WER and the other is CER. One color and one thin line
belong to each model.

    python evaluation/batch-compare.py evaluation/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    }
)

DATASET_ORDER = ("combined", "common_voice", "fleurs", "ukr_dialects", "cv10_test")
COLORS = (
    "#9b2226",
    "#1d4e89",
    "#2a9d8f",
    "#e09f3e",
    "#6a4c93",
    "#386641",
    "#bc4749",
    "#0077b6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Directory whose subfolders are evaluation runs.")
    return parser.parse_args()


def load_runs(folder: Path) -> list[tuple[str, dict[str, dict[str, float]]]]:
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    runs = []
    for path in sorted(item for item in folder.iterdir() if item.is_dir()):
        results = path / "results.json"
        if not results.is_file():
            continue
        payload = json.loads(results.read_text(encoding="utf-8"))
        scored = payload.get("test_sets")
        if not isinstance(scored, dict):
            continue
        rates: dict[str, dict[str, float]] = {}
        for name, metrics in scored.items():
            if not isinstance(metrics, dict):
                continue
            values = {
                metric: float(metrics[metric]) * 100
                for metric in ("wer", "cer")
                if metric in metrics
            }
            if values:
                rates[name] = values
        if rates:
            runs.append((path.name, rates))
    if not runs:
        raise SystemExit(f"No results.json with test set error rates in {folder}")
    return runs


def dataset_names(runs: list[tuple[str, dict[str, dict[str, float]]]]) -> list[str]:
    present = {name for _model, rates in runs for name in rates}
    ordered = [name for name in DATASET_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def write_chart(folder: Path, runs: list[tuple[str, dict[str, dict[str, float]]]]) -> Path:
    datasets = dataset_names(runs)
    positions = list(range(len(datasets)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, len(datasets) * 2.6), 5.8),
        sharex=True,
    )
    for axis, metric, title in zip(axes, ("wer", "cer"), ("WER (%)", "CER (%)")):
        for index, (model, rates) in enumerate(runs):
            xs = [
                position
                for position, name in zip(positions, datasets)
                if metric in rates.get(name, {})
            ]
            ys = [rates[datasets[position]][metric] for position in xs]
            axis.plot(
                xs,
                ys,
                color=COLORS[index % len(COLORS)],
                linewidth=1.0,
                marker="o",
                markersize=7,
                label=model,
            )
        axis.set_xticks(positions, datasets)
        axis.set_xlabel("dataset")
        axis.set_ylabel(title)
        axis.set_ylim(bottom=0)
    axes[0].legend()
    figure.suptitle(folder.name)
    figure.tight_layout()
    destination = folder / "model_comparison.png"
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    destination = write_chart(folder, load_runs(folder))
    print(destination)


if __name__ == "__main__":
    main()
