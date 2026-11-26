"""Share of each judge mark (0-5) within fixed-width teacher-confidence bins.

Takes the same data as ``judge_vs_confidence.py`` (newest ``label_marks.csv``,
confidence from the matching ``labels.csv``, slices with identical text only).
Bins have fixed edges, not equal counts, so sparse low-confidence ranges stay
visible: ``--step`` wide up to ``--fine-from``, ``--fine-step`` wide above it,
where most slices are. The lower panel shows how many slices each bin holds.
Bins with fewer than ``--min-count`` slices are drawn hatched.

Saved next to this script as ``marks_by_confidence_bin_<marks run>.png``.

    python pseudo-labelling/confidences/marks_by_confidence_bin.py
    python pseudo-labelling/confidences/marks_by_confidence_bin.py --step 0.1 --fine-from 0.95 --fine-step 0.005
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from judge_vs_confidence import HERE, MARK_COLORS, confidence_labels, load, newest_marks


def bin_edges(step: float, fine_from: float, fine_step: float) -> np.ndarray:
    coarse = np.arange(0.0, fine_from, step)
    fine = np.arange(fine_from, 1.0, fine_step)
    # Rounding keeps labels such as 0.9 from turning into 0.8999999.
    return np.unique(np.round(np.concatenate([coarse, fine, [1.0]]), 6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marks", type=Path, help="label_marks.csv to use (default: the newest one)")
    parser.add_argument("--confidence", type=Path, help="labels.csv with the confidence column")
    parser.add_argument("--any-text", action="store_true", help="keep slices whose text differs between the two runs")
    parser.add_argument("--step", type=float, default=0.05, help="bin width below --fine-from")
    parser.add_argument("--fine-from", type=float, default=0.9, help="where the narrow bins start")
    parser.add_argument("--fine-step", type=float, default=0.01, help="bin width from --fine-from to 1")
    parser.add_argument("--min-count", type=int, default=100, help="bins with fewer slices are hatched")
    args = parser.parse_args()

    marks_path = (args.marks or newest_marks()).resolve()
    labels_path = (args.confidence or confidence_labels(marks_path)).resolve()
    frame, _ = load(marks_path, labels_path, args.any_text)
    if frame.empty:
        raise SystemExit("Nothing left to compare")

    edges = bin_edges(args.step, args.fine_from, args.fine_step)
    bins = pd.cut(frame["confidence"].clip(0.0, 1.0), edges, include_lowest=True)
    shares = pd.crosstab(bins, frame["mark"], normalize="index").reindex(columns=range(6), fill_value=0.0)
    counts = bins.value_counts().reindex(shares.index)
    intervals = list(shares.index)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(max(10, 0.45 * len(intervals) + 3), 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    positions = np.arange(len(intervals))
    sparse = counts.to_numpy() < args.min_count
    stack = np.zeros(len(intervals))
    for mark in range(6):
        values = shares[mark].to_numpy()
        bars = top.bar(positions, values, bottom=stack, width=0.9, color=MARK_COLORS[mark], label=str(mark))
        for bar, thin in zip(bars, sparse):
            if thin:
                bar.set_hatch("///")
                bar.set_alpha(0.6)
        stack += values
    top.set_ylim(0, 1)
    top.set_ylabel("share of slices")
    top.set_title(f"Judge marks within each confidence bin, {marks_path.parent.name} (n={len(frame):,})", pad=36)
    top.legend(title="mark", ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, title_fontsize=8, frameon=False)

    bottom.bar(positions, counts.to_numpy(), width=0.9, color="#888888")
    bottom.set_yscale("log")
    bottom.set_ylabel("slices (log)")
    bottom.grid(axis="y", alpha=0.3)
    bottom.set_xticks(positions, [f"{max(interval.left, 0.0):.2f}–{interval.right:.2f}" for interval in intervals], rotation=60, ha="right", fontsize=8)
    bottom.set_xlabel(f"teacher confidence bin (hatched: fewer than {args.min_count} slices)")

    figure.tight_layout()
    destination = HERE / f"marks_by_confidence_bin_{marks_path.parent.name}.png"
    figure.savefig(destination, dpi=150)
    print(f"{len(intervals)} bins, {len(frame):,} slices -> {destination}")


if __name__ == "__main__":
    main()
