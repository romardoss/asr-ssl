"""Histogram of the teacher's ``confidence`` from the newest pseudo-labelling ``labels.csv``.

Searches ``pseudo_labelling.output_dir`` (``output/pseudo-labelling``) for
``*/labels.csv`` and takes the one modified last. The plot is saved next to
this script as ``confidence_<run folder>.png``.

    python pseudo-labelling/confidences/plot_confidences.py
    python pseudo-labelling/confidences/plot_confidences.py --labels path/to/labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT_DIR = ROOT / "output" / "pseudo-labelling"


def latest_labels(output_dir: Path) -> Path:
    candidates = [path for path in output_dir.glob("*/labels.csv") if path.is_file()]
    if not candidates:
        raise SystemExit(f"No labels.csv under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, help="labels.csv to plot (default: the newest one)")
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    labels = args.labels or latest_labels(OUTPUT_DIR)
    # A few bytes of a copied labels.csv can be damaged; such rows lose only their bad characters.
    frame = pd.read_csv(labels, usecols=["confidence"], dtype=str, encoding_errors="replace")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").dropna().to_numpy()
    if confidence.size == 0:
        raise SystemExit(f"No confidence values in {labels}")

    run = labels.parent.name
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(confidence, bins=args.bins, range=(0.0, 1.0), color="#3b6ea8", edgecolor="white", linewidth=0.3)
    median, mean = np.median(confidence), confidence.mean()
    axis.axvline(median, color="#c0392b", linestyle="--", linewidth=1, label=f"median {median:.3f}")
    axis.axvline(mean, color="#c0392b", linestyle=":", linewidth=1, label=f"mean {mean:.3f}")
    axis.set_title(f"Teacher confidence, {run} (n={confidence.size:,})")
    axis.set_xlabel("confidence = exp(avg_logprob)")
    axis.set_ylabel("slices")
    axis.set_xlim(0, 1)
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    destination = HERE / f"confidence_{run}.png"
    figure.savefig(destination, dpi=150)
    print(f"{labels} -> {destination} ({confidence.size} values, {len(frame) - confidence.size} empty)")


if __name__ == "__main__":
    main()
