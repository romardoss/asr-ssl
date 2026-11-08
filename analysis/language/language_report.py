"""Build the language report from ``predictions.csv`` of ``detect_languages.py``.

Reads only the CSV, so it needs neither the models nor the audio. Writes to
``analysis/language``:

- ``report.md``: tables per dataset/split, the plots, and examples of clips that
  neither model hears as Ukrainian;
- ``plots/agreement.png``: per dataset/split, how many clips both models, only
  one, or neither call Ukrainian;
- ``plots/other_whisper.png``, ``plots/other_mms.png``: which other languages
  each model hears, as a share of the split;
- ``plots/confidence.png``: the lower of the two Ukrainian probabilities;
- ``plots/duration.png``: share of clips not Ukrainian for both, by clip length.

    python analysis/language/language_report.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "fine-tune") not in sys.path:
    sys.path.insert(0, str(ROOT / "fine-tune"))

from dataset_roles import ROLES  # noqa: E402

REPORT_NAME = "report.md"
PLOTS_DIRNAME = "plots"
SPLIT_ORDER = ("train", "dev", "test")
EXAMPLES = 10
OTHER_LANGUAGES = 8
TEXT_WIDTH = 90

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validated categorical slots 1-4 (light)
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

CATEGORIES = (
    ("both", "обидві: українська"),
    ("whisper_only", "лише Whisper: українська"),
    ("mms_only", "лише MMS: українська"),
    ("neither", "жодна: не українська"),
)
CONFIDENCE_BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0000001]
CONFIDENCE_LABELS = ["<0.5", "0.5–0.8", "0.8–0.9", "0.9–0.95", "0.95–0.99", "≥0.99"]
DURATION_BINS = [0, 1, 2, 3, 5, 10, 20, 30, np.inf]
DURATION_LABELS = ["<1 с", "1–2 с", "2–3 с", "3–5 с", "5–10 с", "10–20 с", "20–30 с", "≥30 с"]

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
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--output", type=Path, default=HERE, help="Folder for report.md and plots/.")
    return parser.parse_args()


# ---------------------------------------------------------------- data


def unit_key(unit: str) -> tuple[int, int, str]:
    dataset, _, split = unit.partition("/")
    order = list(ROLES)
    return (
        order.index(dataset) if dataset in order else len(order),
        SPLIT_ORDER.index(split) if split in SPLIT_ORDER else len(SPLIT_ORDER),
        unit,
    )


def load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run detect_languages.py first.")
    frame = pd.read_csv(path, dtype={"id": str, "error": str, "text": str}, keep_default_na=False, na_values={})
    # A clip tried again after an error has a newer row further down.
    frame = frame.drop_duplicates(["dataset", "split", "id"], keep="last").copy()
    frame["unit"] = frame["dataset"] + "/" + frame["split"]
    for column in ("seconds", "whisper_probability", "whisper_uk", "whisper_ru", "mms_probability", "mms_ukr", "mms_rus"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ok"] = frame["error"].fillna("") == ""
    whisper_uk = frame["whisper_language"] == "uk"
    mms_uk = frame["mms_language"] == "ukr"
    frame["category"] = np.select(
        [whisper_uk & mms_uk, whisper_uk & ~mms_uk, ~whisper_uk & mms_uk],
        ["both", "whisper_only", "mms_only"],
        default="neither",
    )
    frame.loc[~frame["ok"], "category"] = ""
    frame["ukrainian_min"] = frame[["whisper_uk", "mms_ukr"]].min(axis=1, skipna=False)
    return frame


def units_of(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["unit"].unique(), key=unit_key)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit in units_of(frame):
        clips = frame[frame["unit"] == unit]
        ok = clips[clips["ok"]]
        counts = ok["category"].value_counts()
        rows.append(
            {
                "unit": unit,
                "clips": len(clips),
                "errors": int((~clips["ok"]).sum()),
                "checked": len(ok),
                **{key: int(counts.get(key, 0)) for key, _ in CATEGORIES},
                "median_seconds": float(ok["seconds"].median()) if len(ok) else float("nan"),
                "other_whisper": top_other(ok["whisper_language"], "uk"),
                "other_mms": top_other(ok["mms_language"], "ukr"),
            }
        )
    return pd.DataFrame(rows)


def top_other(languages: pd.Series, ukrainian: str, limit: int = 3) -> str:
    counts = languages[languages != ukrainian].value_counts().head(limit)
    return ", ".join(f"{language} {count}" for language, count in counts.items()) or "—"


# ---------------------------------------------------------------- plots


def save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}", flush=True)
    return path


def tidy(axis) -> None:
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=0)


def plot_agreement(table: pd.DataFrame, path: Path) -> Path:
    units = list(table["unit"])
    figure, axis = plt.subplots(figsize=(11, 0.5 * len(units) + 1.8))
    left = np.zeros(len(units))
    checked = table["checked"].clip(lower=1).to_numpy()
    positions = np.arange(len(units))
    for (key, label), color in zip(CATEGORIES, SERIES):
        shares = 100 * table[key].to_numpy() / checked
        axis.barh(positions, shares, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left += shares
    for position, row in zip(positions, table.itertuples(index=False)):
        share = 100 * row.both / max(row.checked, 1)
        axis.text(101, position, f"{share:.1f} % українська · {row.checked:,} кліпів".replace(",", " "), va="center",
                  fontsize=9, color=TEXT_SECONDARY)
    axis.set_yticks(positions, units)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("частка кліпів спліту (%)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    tidy(axis)
    axis.set_title("Мова кліпів за Whisper і MMS", loc="left", pad=34)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handlelength=1.2)
    return save(figure, path)


def heatmap(matrix: pd.DataFrame, title: str, xlabel: str, path: Path, cell_format) -> Path:
    cmap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    values = matrix.to_numpy(dtype=float)
    shown = np.where(values > 0, values, np.nan)
    figure, axis = plt.subplots(figsize=(1.0 * matrix.shape[1] + 3.5, 0.48 * matrix.shape[0] + 1.8))
    peak = np.nanmax(shown) if np.isfinite(shown).any() else 1.0
    image = axis.imshow(shown, cmap=cmap, vmin=0, vmax=peak, aspect="auto")
    for (row, column), value in np.ndenumerate(values):
        if value > 0:
            dark = value > 0.55 * peak
            axis.text(column, row, cell_format(value), ha="center", va="center", fontsize=8.5,
                      color="#ffffff" if dark else TEXT)
    axis.set_xticks(range(matrix.shape[1]), matrix.columns)
    axis.set_yticks(range(matrix.shape[0]), matrix.index)
    axis.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0]), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", pad=12)
    bar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    bar.set_label("частка кліпів спліту (%)", color=TEXT_SECONDARY)
    return save(figure, path)


def plot_other(frame: pd.DataFrame, column: str, ukrainian: str, model: str, path: Path) -> Path | None:
    ok = frame[frame["ok"] & (frame[column] != ukrainian)]
    if ok.empty:
        return None
    languages = list(ok[column].value_counts().head(OTHER_LANGUAGES).index)
    units = units_of(frame)
    totals = frame[frame["ok"]].groupby("unit").size()
    counts = ok[ok[column].isin(languages)].groupby(["unit", column]).size().unstack(fill_value=0)
    matrix = counts.reindex(index=units, columns=languages, fill_value=0).div(totals.reindex(units), axis=0) * 100
    return heatmap(
        matrix.fillna(0),
        f"{model}: які мови, крім української",
        f"мова за {model} (найчастіші {len(languages)})",
        path,
        lambda value: f"{value:.1f}" if value >= 0.1 else "<0.1",
    )


def plot_confidence(frame: pd.DataFrame, path: Path) -> Path | None:
    ok = frame[frame["ok"] & frame["ukrainian_min"].notna()]
    if ok.empty:
        return None
    buckets = pd.cut(ok["ukrainian_min"], bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS, right=False)
    counts = pd.crosstab(ok["unit"], buckets).reindex(columns=CONFIDENCE_LABELS, fill_value=0)
    units = [unit for unit in units_of(frame) if unit in counts.index]
    matrix = counts.reindex(units).div(counts.reindex(units).sum(axis=1), axis=0) * 100
    return heatmap(
        matrix,
        "Наскільки моделі впевнені, що це українська",
        "менша з двох ймовірностей української: Whisper P(uk) і MMS P(ukr)",
        path,
        lambda value: f"{value:.1f}" if value >= 0.1 else "<0.1",
    )


def plot_duration(frame: pd.DataFrame, path: Path) -> Path | None:
    ok = frame[frame["ok"] & frame["seconds"].notna()]
    if ok.empty:
        return None
    buckets = pd.cut(ok["seconds"], bins=DURATION_BINS, labels=DURATION_LABELS, right=False)
    grouped = ok.groupby(buckets, observed=False)["category"]
    clips = grouped.size()
    share = 100 * grouped.apply(lambda values: (values != "both").sum()) / clips.where(clips > 0)
    figure, axis = plt.subplots(figsize=(10, 4.6))
    positions = np.arange(len(DURATION_LABELS))
    axis.bar(positions, share.fillna(0).to_numpy(), width=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for position, value, count in zip(positions, share.to_numpy(), clips.to_numpy()):
        if count:
            axis.text(position, value, f"{value:.1f} %\n{count:,} кл.".replace(",", " "), ha="center", va="bottom",
                      fontsize=8.5, color=TEXT_SECONDARY)
    axis.set_xticks(positions, DURATION_LABELS)
    axis.set_ylabel("не українська хоча б за однією моделлю (%)")
    axis.set_xlabel("тривалість кліпу (усі датасети разом)")
    axis.set_ylim(0, max(5.0, float(np.nanmax(share.to_numpy())) * 1.25 if share.notna().any() else 5.0))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    tidy(axis)
    axis.set_title("Частка кліпів, не українських хоча б за однією моделлю, за тривалістю", loc="left", pad=12)
    return save(figure, path)


# ---------------------------------------------------------------- report


def spaced(number) -> str:
    return f"{int(number):,}".replace(",", " ")


def share(part, total) -> str:
    return f"{100 * part / total:.1f} %" if total else "—"


def cell(text) -> str:
    value = " ".join(str(text).split()).replace("|", "\\|")
    return value if len(value) <= TEXT_WIDTH else value[: TEXT_WIDTH - 1] + "…"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return lines


def probability(value) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def git_commit() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return "—"
    return completed.stdout.strip() or "—"


def build_report(predictions: Path, folder: Path) -> Path:
    frame = load(predictions)
    if frame.empty:
        raise SystemExit(f"{predictions} has no rows.")
    stats = summary(frame)
    plots_dir = folder / PLOTS_DIRNAME
    plots = [
        plot_agreement(stats, plots_dir / "agreement.png"),
        plot_other(frame, "whisper_language", "uk", "Whisper", plots_dir / "other_whisper.png"),
        plot_other(frame, "mms_language", "ukr", "MMS", plots_dir / "other_mms.png"),
        plot_confidence(frame, plots_dir / "confidence.png"),
        plot_duration(frame, plots_dir / "duration.png"),
    ]
    plots = [path for path in plots if path is not None]

    checked = int(stats["checked"].sum())
    both = int(stats["both"].sum())
    lines = [
        "# Мова кліпів у датасетах",
        "",
        f"Створено {datetime.now():%Y-%m-%d %H:%M}, коміт `{git_commit()}`. "
        "Скрипти: `analysis/language/detect_languages.py` (моделі) і `analysis/language/language_report.py` (звіт). "
        "Логіка та сама, що відбирає українські кліпи VOA: Whisper medium (faster-whisper, перші 120 с, VAD) "
        "і MMS `facebook/mms-lid-256` (перші 300 с, argmax). Кліп вважається українським, лише якщо Whisper каже `uk`, "
        "а MMS — `ukr`. Коди мов: Whisper — ISO 639-1 (`uk`, `ru`), MMS — ISO 639-3 (`ukr`, `rus`).",
        "",
        "## Головне",
        "",
    ]
    summary_lines = [
        f"Перевірено {spaced(checked)} кліпів у {len(stats)} сплітах; обидві моделі чують українську в "
        f"{spaced(both)} ({share(both, checked)}).",
    ]
    errors = int(stats["errors"].sum())
    if errors:
        summary_lines.append(f"Помилок декодування чи моделей: {spaced(errors)}. Вони не входять у частки, див. розділ 4.")
    stats["not_both"] = stats["checked"] - stats["both"]
    worst = stats[stats["checked"] > 0].assign(rate=lambda table_: table_["not_both"] / table_["checked"])
    worst = worst.sort_values("rate", ascending=False).head(3)
    summary_lines.append(
        "Найбільша частка не українською хоча б за однією моделлю: "
        + "; ".join(f"`{row.unit}` {share(row.not_both, row.checked)}" for row in worst.itertuples(index=False))
        + "."
    )
    neither = int(stats["neither"].sum())
    summary_lines.append(
        f"Обидві моделі не чують українську в {spaced(neither)} кліпах ({share(neither, checked)}). "
        "Саме їх варто прослухати насамперед (розділ 3)."
    )
    lines += [f"{number}. {text}" for number, text in enumerate(summary_lines, start=1)]

    lines += ["", "## 1. Мова за сплітами", ""]
    lines += [
        "Частки — від кліпів без помилок. «Інші мови» — найчастіші не українські відповіді кожної моделі з кількістю кліпів.",
        "",
    ]
    lines += table(
        ["Спліт", "Кліпів", "Обидві: укр.", "Лише Whisper: укр.", "Лише MMS: укр.", "Жодна", "Медіана, с", "Інші мови (Whisper)", "Інші мови (MMS)"],
        [
            [
                row.unit,
                spaced(row.checked),
                f"{spaced(row.both)} ({share(row.both, row.checked)})",
                f"{spaced(row.whisper_only)} ({share(row.whisper_only, row.checked)})",
                f"{spaced(row.mms_only)} ({share(row.mms_only, row.checked)})",
                f"{spaced(row.neither)} ({share(row.neither, row.checked)})",
                "—" if pd.isna(row.median_seconds) else f"{row.median_seconds:.1f}",
                row.other_whisper,
                row.other_mms,
            ]
            for row in stats.itertuples(index=False)
        ],
    )
    lines += [
        "",
        "Як читати:",
        "",
        "1. «Лише Whisper» чи «лише MMS» — моделі розходяться. Це може бути помилка однієї з моделей, а не інша мова: для коротких кліпів і діалектів визначення мови менш надійне (див. графік тривалості).",
        "2. «Жодна» — кандидати на іншу мову, суржик, тишу чи шум. Що саме, видно лише після прослуховування (розділ 3).",
        "3. У VOA такі кліпи відкидаються з навчання. У цих датасетах нічого не відкидається: звіт лише показує, скільки їх.",
    ]

    lines += ["", "## 2. Графіки", ""]
    for path in plots:
        lines += [f"![{path.stem}]({path.relative_to(folder).as_posix()})", ""]

    lines += ["## 3. Приклади: жодна модель не чує українську", ""]
    lines.append(
        f"До {EXAMPLES} кліпів на спліт з найменшою ймовірністю української (менша з Whisper P(uk) і MMS P(ukr)). "
        "ID — той самий, що в навчанні й `predictions.csv`."
    )
    for unit in units_of(frame):
        chosen = frame[(frame["unit"] == unit) & (frame["category"] == "neither")]
        if chosen.empty:
            continue
        chosen = chosen.sort_values("ukrainian_min", na_position="last").head(EXAMPLES)
        lines += ["", f"### `{unit}`", ""]
        lines += table(
            ["ID", "с", "Whisper", "MMS (топ-3)", "Текст"],
            [
                [
                    row.id,
                    "—" if pd.isna(row.seconds) else f"{row.seconds:.1f}",
                    f"{row.whisper_language} {probability(row.whisper_probability)}",
                    row.mms_top,
                    row.text,
                ]
                for row in chosen.itertuples(index=False)
            ],
        )

    failed = frame[~frame["ok"]]
    lines += ["", "## 4. Помилки", ""]
    if failed.empty:
        lines.append("Помилок немає.")
    else:
        lines.append(
            f"{spaced(len(failed))} кліпів. Повторний запуск `detect_languages.py` пробує їх знову. Перші {EXAMPLES}:"
        )
        lines += [""] + table(
            ["Спліт", "ID", "Помилка"],
            [[row.unit, row.id, row.error] for row in failed.head(EXAMPLES).itertuples(index=False)],
        )

    lines += [
        "",
        "## 5. Файли",
        "",
        "1. `predictions.csv` — один рядок на кліп: мова й ймовірність за Whisper, P(uk) і P(ru) за Whisper, "
        "мова, P(ukr), P(rus) і топ-3 за MMS, `ukrainian` (обидві моделі), тривалість, текст, помилка.",
        "2. `plots/` — графіки з розділу 2.",
        "",
    ]
    path = folder / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    print(f"report: {build_report(args.predictions, args.output)}", flush=True)


if __name__ == "__main__":
    main()
