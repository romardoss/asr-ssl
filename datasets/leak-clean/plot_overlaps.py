"""Plot the overlaps found by find_overlaps.py and the coverage check.

Reads ``duplicates.csv``, ``units.csv``, and ``coverage.csv`` from
``output/leak-clean`` and writes to its ``plots/`` folder:

- ``overlap_matrix.png``: how many distinct normalized texts each pair of
  dataset/split shares. The diagonal counts texts repeated inside one split.
- ``eval_leaks.png``: share of each dev and test split whose text is also in
  any train split, for every text and for texts of at least 4 words.
- ``coverage.png``: audio without metadata, metadata without audio, and audio
  without text. Not written when nothing was found.

    python datasets/leak-clean/plot_overlaps.py
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from leak_common import COVERAGE_NAME, DUPLICATES_NAME, PLOTS_DIRNAME, UNITS_NAME, output_dir, unit_key

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834")
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
LONG_TEXT_WORDS = 4

PROBLEM_LABELS = {
    "audio_without_metadata": "аудіо без метаданих",
    "metadata_without_audio": "метадані без аудіо",
    "audio_without_text": "аудіо без тексту",
    "duplicate_audio_name": "однакові імена аудіо",
}
SKIPPED_LABELS = {
    "dataset_not_found": "не знайдено",
    "audio_not_checked": "аудіо не перевірено",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlesize": 14,
        "axes.titlecolor": TEXT,
        "axes.labelsize": 11,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": TEXT,
        "legend.fontsize": 10,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Folder with the CSV files. Default: output/leak-clean.")
    return parser.parse_args()


def spaced(number: float) -> str:
    return f"{int(number):,}".replace(",", " ")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run collect_metadata.py and find_overlaps.py first.")
    return pd.read_csv(path, dtype={"split": str, "id": str, "text_norm": str}, keep_default_na=False)


def save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def overlap_counts(duplicates: pd.DataFrame, units: list[str]) -> np.ndarray:
    index = {unit: position for position, unit in enumerate(units)}
    counts = np.zeros((len(units), len(units)), dtype=np.int64)
    per_group = duplicates.drop_duplicates("group")
    for listed in per_group["units"]:
        present = [unit for unit in listed.split("; ") if unit in index]
        for first, second in combinations(present, 2):
            counts[index[first], index[second]] += 1
            counts[index[second], index[first]] += 1
    duplicates = duplicates.assign(unit=duplicates["dataset"] + "/" + duplicates["split"])
    repeated = duplicates.groupby(["group", "unit"]).size()
    for (_group, unit), copies in repeated.items():
        if copies > 1 and unit in index:
            counts[index[unit], index[unit]] += 1
    return counts


def plot_matrix(duplicates: pd.DataFrame, units: list[str], path: Path) -> Path:
    counts = overlap_counts(duplicates, units)
    size = max(6.5, 0.62 * len(units) + 2.5)
    figure, axis = plt.subplots(figsize=(size + 1.2, size))
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    colormap.set_bad(SURFACE)
    masked = np.ma.masked_less(counts, 1)
    peak = max(int(counts.max()), 2)
    image = axis.imshow(masked, cmap=colormap, norm=LogNorm(vmin=1, vmax=peak))

    for row in range(len(units)):
        for column in range(len(units)):
            value = counts[row, column]
            if value < 1:
                continue
            shade = np.log(value) / np.log(peak) if peak > 1 else 0
            axis.text(
                column,
                row,
                spaced(value),
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if shade > 0.5 else TEXT,
            )

    axis.set_xticks(range(len(units)), units, rotation=45, ha="right")
    axis.set_yticks(range(len(units)), units)
    axis.set_xticks(np.arange(-0.5, len(units)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(units)), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.8)
    axis.tick_params(which="minor", length=0)
    axis.tick_params(which="major", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    datasets = [unit.split("/")[0] for unit in units]
    for position in range(1, len(units)):
        if datasets[position] != datasets[position - 1]:
            axis.axhline(position - 0.5, color=TEXT_SECONDARY, linewidth=1.2)
            axis.axvline(position - 0.5, color=TEXT_SECONDARY, linewidth=1.2)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("різних текстів (логарифмічна шкала)")
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: spaced(value)))
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    axis.set_title("Спільні нормалізовані тексти між сплітами", loc="left", pad=28)
    axis.text(
        0,
        1.015,
        "Клітинка: скільки різних текстів є в обох сплітах. Діагональ: тексти, що повторюються всередині спліту.",
        transform=axis.transAxes,
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    return save(figure, path)


def plot_eval_leaks(duplicates: pd.DataFrame, units: pd.DataFrame, path: Path) -> Path | None:
    evaluated = [unit for unit in units["unit"] if not unit.endswith("/train")]
    if not evaluated:
        return None
    totals = dict(zip(units["unit"], units["clips"]))
    leaked = duplicates[duplicates["split"] != "train"]
    leaked = leaked[leaked["units"].str.contains("/train", regex=False)]
    leaked = leaked.assign(unit=leaked["dataset"] + "/" + leaked["split"])
    every = leaked.groupby("unit").size()
    long = leaked[leaked["words"].astype(int) >= LONG_TEXT_WORDS].groupby("unit").size()

    figure, axis = plt.subplots(figsize=(9.5, 0.75 * len(evaluated) + 1.8))
    positions = np.arange(len(evaluated))
    height = 0.36
    series = (
        ("усі тексти", every, -height / 2),
        (f"тексти від {LONG_TEXT_WORDS} слів", long, height / 2),
    )
    peak = 0.0
    for (label, counts, offset), color in zip(series, SERIES):
        shares = [100 * counts.get(unit, 0) / totals[unit] for unit in evaluated]
        peak = max([peak, *shares])
        bars = axis.barh(positions + offset, shares, height=height, color=color, label=label)
        for bar, unit, share in zip(bars, evaluated, shares):
            axis.annotate(
                f"{share:.1f}%  ({spaced(counts.get(unit, 0))} з {spaced(totals[unit])})",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=TEXT,
            )
    axis.set_yticks(positions, evaluated)
    axis.invert_yaxis()
    axis.set_xlim(0, max(peak * 1.45, 5))
    axis.set_xticks([tick for tick in axis.get_xticks() if 0 <= tick <= 100])
    axis.set_xlabel("частка кліпів, текст яких є в train (%)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="lower right")
    axis.set_title("Dev і test: тексти, які є в будь-якому train", loc="left")
    return save(figure, path)


def plot_coverage(coverage: pd.DataFrame, path: Path) -> Path | None:
    found = coverage[(coverage["count"].astype(int) > 0) & coverage["problem"].isin(PROBLEM_LABELS)].copy()
    skipped = coverage[coverage["problem"].isin(SKIPPED_LABELS)]
    if found.empty:
        if path.is_file():
            path.unlink()
        return None
    found["label"] = [
        f"{dataset}/{split}: {PROBLEM_LABELS[problem]}" if split else f"{dataset}: {PROBLEM_LABELS[problem]}"
        for dataset, split, problem in zip(found["dataset"], found["split"], found["problem"])
    ]
    found["rank"] = [unit_key(f"{dataset}/{split}") for dataset, split in zip(found["dataset"], found["split"])]
    found = found.sort_values("rank")
    counts = found["count"].astype(int).tolist()

    figure, axis = plt.subplots(figsize=(9.5, 0.5 * len(found) + 1.8))
    positions = np.arange(len(found))
    bars = axis.barh(positions, counts, height=0.5, color=SERIES[0])
    for bar, count in zip(bars, counts):
        axis.annotate(
            spaced(count),
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    axis.set_yticks(positions, found["label"])
    axis.invert_yaxis()
    axis.set_xlim(0, max(counts) * 1.2)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: spaced(value)))
    axis.set_xlabel("кількість")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.set_title("Аудіо й метадані, що не мають пари", loc="left")
    if not skipped.empty:
        names = sorted(
            {f"{dataset}/{split}" if split else dataset for dataset, split in zip(skipped["dataset"], skipped["split"])},
            key=unit_key,
        )
        figure.text(0.01, -0.02, "Не перевірено: " + ", ".join(names), fontsize=9, color=TEXT_SECONDARY)
    return save(figure, path)


def main() -> None:
    args = parse_args()
    folder = output_dir(args.output)
    duplicates = read_csv(folder / DUPLICATES_NAME)
    units = read_csv(folder / UNITS_NAME)
    coverage = read_csv(folder / COVERAGE_NAME)
    plots = folder / PLOTS_DIRNAME

    unit_names = sorted(units["unit"], key=unit_key)
    written = [plot_matrix(duplicates, unit_names, plots / "overlap_matrix.png")]
    written.append(plot_eval_leaks(duplicates, units, plots / "eval_leaks.png"))
    coverage_plot = plot_coverage(coverage, plots / "coverage.png")
    if coverage_plot is None:
        print("coverage: no audio/metadata mismatch found, no coverage plot")
    skipped = coverage[coverage["problem"].isin(SKIPPED_LABELS)]
    for dataset, split, problem in zip(skipped["dataset"], skipped["split"], skipped["problem"]):
        where = f"{dataset}/{split}" if split else dataset
        print(f"coverage: {where} {problem}, so it was not checked")
    written.append(coverage_plot)
    for path in written:
        if path is not None:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
