"""Compare the LLM judge's marks (0-5) with the teacher's own confidence.

Marks come from the newest ``label_marks.csv`` under ``output/pseudo-labelling``
(``--marks`` overrides). Its ``index`` points to a row of ``labels.csv`` in the
same folder. Confidence comes from that ``labels.csv`` when it has the column;
otherwise from the newest ``labels.csv`` of the same teacher that has it
(``--confidence`` overrides). The two files are joined on ``ID``. When they
are different runs, only slices whose text is identical in both are kept, so
each mark was given to exactly the text whose confidence is used
(``--any-text`` keeps all of them).

Writes next to this script:

1. ``judge_vs_confidence_<marks run>.png``: confidence per mark, mean mark and
   mark shares per confidence bin, and what a confidence threshold keeps;
2. ``judge_vs_confidence_<marks run>.md``: the numbers.

    python pseudo-labelling/confidences/judge_vs_confidence.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT_DIR = ROOT / "output" / "pseudo-labelling"
MARKS_NAME = "label_marks.csv"
LABELS_NAME = "labels.csv"
NUMERIC = ["confidence", "avg_logprob", "min_logprob", "num_tokens", "compression_ratio", "truncated"]
GOOD_MARK = 4
BAD_MARK = 1
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98]
MARK_COLORS = ["#7f1d1d", "#c2410c", "#eab308", "#a3c45a", "#4d9a4f", "#1f5f3a"]


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    # A few bytes of a copied file can be damaged; such rows lose only their bad characters.
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, encoding_errors="replace")


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def newest_marks() -> Path:
    candidates = [path for path in OUTPUT_DIR.glob(f"*/{MARKS_NAME}") if path.is_file()]
    if not candidates:
        raise SystemExit(f"No {MARKS_NAME} under {OUTPUT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def teacher_slug(run: Path) -> str:
    """``<teacher>-NN`` -> ``<teacher>``."""
    name, _, number = run.name.rpartition("-")
    return name if number.isdigit() else run.name


def confidence_labels(marks: Path) -> Path:
    own = marks.parent / LABELS_NAME
    if own.is_file() and "confidence" in header(own):
        return own
    slug = teacher_slug(marks.parent)
    candidates = [
        path
        for path in OUTPUT_DIR.glob(f"{slug}-*/{LABELS_NAME}")
        if teacher_slug(path.parent) == slug and "confidence" in header(path)
    ]
    if not candidates:
        raise SystemExit(f"No {LABELS_NAME} with a confidence column for teacher {slug}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Chance that a random positive has a higher score than a random negative (Mann-Whitney)."""
    ranks = rank(score)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load(marks_path: Path, labels_path: Path, any_text: bool) -> tuple[pd.DataFrame, dict]:
    marks = read_csv(marks_path, ["index", "ID", "mark"])
    marks = marks[marks["mark"].str.fullmatch(r"[0-5]") & marks["index"].str.fullmatch(r"\d+")]
    marks = marks.drop_duplicates("index", keep="last")
    marked_labels = read_csv(marks_path.parent / LABELS_NAME, ["ID", "text"]).reset_index(names="row")
    marks = marks.assign(row=marks["index"].astype(int)).merge(marked_labels, on="row", suffixes=("", "_labels"))
    # A mark belongs to the labels row it indexes; skip rows whose ID disagrees.
    marks = marks[marks["ID"] == marks["ID_labels"]]

    scored = read_csv(labels_path, ["ID", "text", *NUMERIC])
    joined = marks[["ID", "mark", "text"]].merge(scored, on="ID", suffixes=("_marked", ""))
    info = {"marks": len(marks), "joined": len(joined)}
    if labels_path.parent != marks_path.parent and not any_text:
        joined = joined[joined["text_marked"] == joined["text"]]
    info["same_text"] = len(joined)
    for column in ["mark", *NUMERIC]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined = joined.dropna(subset=["mark", "confidence"])
    joined["mark"] = joined["mark"].astype(int)
    info["used"] = len(joined)
    return joined.reset_index(drop=True), info


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    mark = frame["mark"].to_numpy()
    good, bad = mark >= GOOD_MARK, mark <= BAD_MARK
    rows = []
    for column in NUMERIC:
        part = frame[column].notna().to_numpy()
        values = frame[column].to_numpy()[part]
        if values.size < 2 or np.unique(values).size < 2:
            continue
        rows.append(
            {
                "field": column,
                "spearman": spearman(values, mark[part]),
                "pearson": float(np.corrcoef(values, mark[part])[0, 1]),
                f"auroc mark>={GOOD_MARK}": auroc(values, good[part]),
                f"auroc mark<={BAD_MARK}": auroc(-values, bad[part]),
            }
        )
    return pd.DataFrame(rows)


def per_mark(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("mark")["confidence"]
    table = grouped.describe(percentiles=[0.1, 0.25, 0.5, 0.75])
    table = table[["count", "mean", "10%", "25%", "50%", "75%"]].rename(columns={"50%": "median"})
    return table.astype({"count": int})


def per_bin(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Equal-count confidence bins: mean mark and the share of each mark."""
    labels = pd.qcut(frame["confidence"], q=bins, duplicates="drop")
    shares = pd.crosstab(labels, frame["mark"], normalize="index").reindex(columns=range(6), fill_value=0.0)
    table = frame.groupby(labels, observed=True).agg(
        low=("confidence", "min"), high=("confidence", "max"), slices=("mark", "size"), mean_mark=("mark", "mean")
    )
    return table.join(shares)


def thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        kept = frame[frame["confidence"] >= threshold]
        rows.append(
            {
                "confidence >=": threshold,
                "kept": len(kept) / len(frame),
                "mean mark": kept["mark"].mean(),
                f"mark>={GOOD_MARK}": (kept["mark"] >= GOOD_MARK).mean(),
                f"mark<={BAD_MARK}": (kept["mark"] <= BAD_MARK).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, bins: pd.DataFrame, cuts: pd.DataFrame, title: str, destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axis = axes[0, 0]
    data = [frame.loc[frame["mark"] == mark, "confidence"].to_numpy() for mark in range(6)]
    present = [mark for mark in range(6) if data[mark].size]
    box = axis.boxplot([data[mark] for mark in present], positions=present, showfliers=False, patch_artist=True)
    for patch, mark in zip(box["boxes"], present):
        patch.set_facecolor(MARK_COLORS[mark])
        patch.set_alpha(0.75)
    axis.set_xticks(present, [f"{mark}\nn={data[mark].size:,}" for mark in present])
    axis.set_xlabel("judge mark")
    axis.set_ylabel("teacher confidence")
    axis.set_title("Confidence per judge mark (no outliers drawn)")
    axis.grid(axis="y", alpha=0.3)

    centers = (bins["low"] + bins["high"]) / 2
    axis = axes[0, 1]
    axis.plot(centers, bins["mean_mark"], marker="o", color="#3b6ea8")
    axis.set_xlabel("confidence (bin middle, equal-count bins)")
    axis.set_ylabel("mean judge mark")
    axis.set_title("Mean judge mark by confidence")
    axis.grid(alpha=0.3)

    axis = axes[1, 0]
    positions = np.arange(len(bins))
    bottom = np.zeros(len(bins))
    for mark in range(6):
        axis.bar(positions, bins[mark], bottom=bottom, color=MARK_COLORS[mark], width=0.9, label=str(mark))
        bottom += bins[mark].to_numpy()
    axis.set_xticks(positions, [f"{low:.3f}" for low in bins["low"]], rotation=60, fontsize=8)
    axis.set_xlabel("confidence bin (lower edge, equal-count bins)")
    axis.set_ylabel("share of slices")
    axis.set_ylim(0, 1)
    axis.set_title("Judge marks within each confidence bin")
    axis.legend(title="mark", ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.08), fontsize=8)
    axis.title.set_position((0.5, 1.18))

    axis = axes[1, 1]
    axis.plot(cuts["confidence >="], cuts["kept"], marker="o", color="#555555", label="slices kept")
    axis.plot(cuts["confidence >="], cuts[f"mark>={GOOD_MARK}"], marker="o", color="#1f5f3a", label=f"mark >= {GOOD_MARK} among kept")
    axis.plot(cuts["confidence >="], cuts[f"mark<={BAD_MARK}"], marker="o", color="#7f1d1d", label=f"mark <= {BAD_MARK} among kept")
    axis.set_xlabel("confidence threshold")
    axis.set_ylabel("share")
    axis.set_ylim(0, 1)
    axis.set_title("Filtering by confidence")
    axis.legend(loc="center left")
    axis.grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def markdown(table: pd.DataFrame, digits: int = 3) -> str:
    columns = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table.itertuples(index=False):
        cells = [f"{value:.{digits}f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marks", type=Path, help=f"{MARKS_NAME} to use (default: the newest one)")
    parser.add_argument("--confidence", type=Path, help=f"{LABELS_NAME} with the confidence column")
    parser.add_argument("--any-text", action="store_true", help="keep slices whose text differs between the two runs")
    parser.add_argument("--bins", type=int, default=20, help="equal-count confidence bins")
    args = parser.parse_args()

    marks_path = (args.marks or newest_marks()).resolve()
    labels_path = (args.confidence or confidence_labels(marks_path)).resolve()
    frame, info = load(marks_path, labels_path, args.any_text)
    if frame.empty:
        raise SystemExit("Nothing left to compare")

    corr = correlations(frame)
    by_mark = per_mark(frame).reset_index()
    by_bin = per_bin(frame, args.bins)
    cuts = thresholds(frame)

    run = marks_path.parent.name
    png = HERE / f"judge_vs_confidence_{run}.png"
    report = HERE / f"judge_vs_confidence_{run}.md"
    plot(frame, by_bin, cuts, f"Judge marks vs teacher confidence, {run} (n={len(frame):,})", png)

    bins_table = by_bin.reset_index(drop=True)
    bins_table.columns = [f"mark {column}" if isinstance(column, int) else column for column in bins_table.columns]
    same_run = labels_path.parent == marks_path.parent
    text_note = (
        "Marks and confidence come from the same run."
        if same_run
        else f"Different runs: {'all joined slices kept' if args.any_text else 'only slices with identical text in both runs are kept'}."
    )
    lines = [
        f"# Judge marks vs teacher confidence: {run}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `pseudo-labelling/confidences/judge_vs_confidence.py`.",
        "",
        f"1. Marks: `{marks_path.relative_to(ROOT)}` ({info['marks']:,} valid marks).",
        f"2. Confidence: `{labels_path.relative_to(ROOT)}`.",
        f"3. Joined on ID: {info['joined']:,}; same text: {info['same_text']:,}; used: {info['used']:,}. {text_note}",
        "",
        "## Correlation with the mark",
        "",
        f"AUROC mark>={GOOD_MARK}: how well a higher value separates marks {GOOD_MARK}-5 from the rest. "
        f"AUROC mark<={BAD_MARK}: how well a lower value finds marks 0-{BAD_MARK}. 0.5 is chance.",
        "",
        markdown(corr),
        "",
        "## Confidence per mark",
        "",
        markdown(by_mark),
        "",
        f"## Marks per confidence bin ({len(by_bin)} equal-count bins)",
        "",
        markdown(bins_table),
        "",
        "## Filtering by a confidence threshold",
        "",
        markdown(cuts),
        "",
        f"![plots]({png.name})",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"marks: {marks_path}\nconfidence: {labels_path}")
    print(f"joined {info['joined']:,}, same text {info['same_text']:,}, used {info['used']:,}")
    print(corr.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"-> {png}\n-> {report}")


if __name__ == "__main__":
    main()
