"""Plot VOA-ukr metadata statistics.

Same figures as notebooks/voa_ukr_meta_stats.ipynb. Reads
datasets/voa-ukr/voa_ukr_meta.json and writes plots to output/eda. A partial checkpoint
is fine.

Run from the repo root:

    python src/meta_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META_PATH = Path(__file__).resolve().parents[2] / "datasets" / "voa-ukr" / "voa_ukr_meta.json"
FIGURE_DIR = ROOT / "output" / "eda"


def load_rows(path: Path) -> tuple[dict, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["files"] if isinstance(payload, dict) else payload
    return payload if isinstance(payload, dict) else {}, pd.DataFrame(rows)


def save_fig(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=120, bbox_inches="tight")
    plt.close()


def summary_table(df: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    hours = ok["duration_seconds"].fillna(0).sum() / 3600 if "duration_seconds" in ok.columns else 0
    labelled_rate = ok["is_labelled"].fillna(False).mean() if "is_labelled" in ok.columns else None
    n_errors = int(df["error"].notna().sum()) if "error" in df.columns else 0
    n_speakers = ok["speaker_id"].nunique(dropna=True) if "speaker_id" in ok.columns else 0
    return pd.DataFrame(
        {
            "value": [
                len(df),
                len(ok),
                n_errors,
                round(hours, 2),
                None if labelled_rate is None else round(float(labelled_rate), 3),
                n_speakers,
                ok["source"].dropna().unique().tolist() if "source" in ok.columns else None,
                ok["split"].dropna().unique().tolist() if "split" in ok.columns else None,
            ]
        },
        index=[
            "records",
            "without_error",
            "errors",
            "hours",
            "labelled_fraction",
            "speaker_ids",
            "source",
            "split",
        ],
    )


def plot_duration(ok: pd.DataFrame) -> None:
    if "duration_seconds" not in ok.columns or ok["duration_seconds"].dropna().empty:
        print("no durations to plot")
        return
    duration_min = ok["duration_seconds"].dropna() / 60
    plt.figure()
    plt.hist(duration_min, bins=60)
    plt.xlabel("duration (minutes)")
    plt.ylabel("clips")
    plt.title("Clip duration")
    save_fig("duration_hist.png")

    cap = duration_min.quantile(0.99)
    plt.figure()
    plt.hist(duration_min[duration_min <= cap], bins=60)
    plt.xlabel("duration (minutes)")
    plt.ylabel("clips")
    plt.title(f"Clip duration up to the 99th percentile ({cap:.1f} min)")
    save_fig("duration_hist_p99.png")

    ordered = ok["duration_seconds"].fillna(0).sort_values()
    plt.figure()
    plt.plot(range(len(ordered)), ordered.cumsum() / 3600)
    plt.xlabel("clips sorted by duration")
    plt.ylabel("cumulative hours")
    plt.title("Cumulative duration")
    save_fig("duration_cumulative.png")


def language_pie(ok: pd.DataFrame, column: str, filename: str) -> None:
    if column not in ok.columns:
        print("missing", column)
        return
    counts = ok[column].fillna("unknown").value_counts()
    plt.figure(figsize=(8, 6))
    wedges, _ = plt.pie(counts.values, startangle=90)
    plt.legend(
        wedges,
        [f"{name} ({count})" for name, count in counts.items()],
        title=column,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.title(column)
    save_fig(filename)


def count_bar(ok: pd.DataFrame, column: str, filename: str) -> None:
    if column not in ok.columns:
        print("missing", column)
        return
    counts = ok[column].fillna("missing").astype(str).value_counts()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("clips")
    plt.title(column)
    save_fig(filename)


def plot_rest(ok: pd.DataFrame) -> None:
    count_bar(ok, "is_labelled", "is_labelled.png")
    count_bar(ok, "sample_rate", "sample_rate.png")
    count_bar(ok, "num_channels", "num_channels.png")
    count_bar(ok, "codec", "codec.png")
    count_bar(ok, "split", "split.png")

    if "speaker_id" in ok.columns:
        known = int(ok["speaker_id"].notna().sum())
        print(f"speaker_id is set on {known} of {len(ok)} clips")

    if "whisper_transcription" in ok.columns:
        lengths = ok["whisper_transcription"].fillna("").str.len()
        plt.figure()
        plt.hist(lengths, bins=60)
        plt.xlabel("whisper transcript characters")
        plt.ylabel("clips")
        plt.title("Whisper transcript length")
        save_fig("whisper_length.png")

    if {"language", "duration_seconds"} <= set(ok.columns):
        top = ok["language"].value_counts().head(8).index
        pairs = []
        for language in top:
            series = ok.loc[ok["language"] == language, "duration_seconds"].dropna() / 60
            if len(series):
                pairs.append((language, series))
        if pairs:
            plt.figure(figsize=(10, 4))
            plt.boxplot(
                [series for _, series in pairs],
                tick_labels=[language for language, _ in pairs],
                showfliers=False,
            )
            plt.ylabel("duration (minutes)")
            plt.title("Duration by Whisper language")
            save_fig("duration_by_language.png")


def main() -> None:
    payload, df = load_rows(META_PATH)
    ok = df[df["error"].isna()] if "error" in df.columns else df
    print(META_PATH)
    print("records", len(df), "updated", payload.get("updated"))
    print(summary_table(df, ok))
    plot_duration(ok)
    language_pie(ok, "language", "language.png")
    plot_rest(ok)
    print("figures", FIGURE_DIR)


if __name__ == "__main__":
    main()
