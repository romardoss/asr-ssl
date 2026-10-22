"""Bar plot of pseudo-label quality marks.

Reads ``label_marks.csv`` and writes ``label_quality.png`` beside it.
If an id was scored more than once, the last mark is kept.

    python pseudo-labelling/llm-as-a-judge/plot_marks.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKS = ROOT / "output" / "pseudo-labelling" / "model-whisper-uk-epoch00-01" / "label_marks.csv"
MARKS = tuple(range(6))
COLORS = ("#9b2226", "#bb3e03", "#ca6702", "#ee9b00", "#90a955", "#4d7c0f")

plt.rcParams.update(
    {
        "figure.figsize": (10, 6.2),
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", type=Path, default=DEFAULT_MARKS, help="label_marks.csv to summarise.")
    return parser.parse_args()


def latest_marks(path: Path) -> Counter[int]:
    if not path.is_file():
        raise SystemExit(f"Marks file not found: {path}")
    latest: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latest[row["ID"]] = int(row["mark"])
    return Counter(latest.values())


def write_plot(counts: Counter[int], path: Path) -> None:
    total = sum(counts[mark] for mark in MARKS)
    values = [counts[mark] for mark in MARKS]
    figure, axis = plt.subplots()
    bars = axis.bar([str(mark) for mark in MARKS], values, color=COLORS, width=0.72)
    for bar, count in zip(bars, values):
        share = 100 * count / total if total else 0
        axis.annotate(
            f"{count:,}\n{share:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axis.set_title(f"Pseudo-label quality  ·  {total:,} labels")
    axis.set_xlabel("mark")
    axis.set_ylabel("count")
    axis.set_ylim(0, max(values) * 1.22 if values else 1)
    axis.grid(axis="x", visible=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    marks_path = args.marks if args.marks.is_absolute() else ROOT / args.marks
    counts = latest_marks(marks_path)
    destination = marks_path.parent / "label_quality.png"
    write_plot(counts, destination)
    total = sum(counts[mark] for mark in MARKS)
    print(f"labels {total}", flush=True)
    for mark in MARKS:
        count = counts[mark]
        share = 100 * count / total if total else 0
        print(f"mark {mark}: {count} ({share:.1f}%)", flush=True)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
