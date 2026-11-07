"""Check every dataset for repeats and leaks and write one readable report.

Runs the three steps in order and then writes ``report.md``:

1. ``collect_metadata.py``: one metadata row per clip, and whether every
   audio has metadata (``metadata.parquet``, ``coverage.csv``);
2. ``find_overlaps.py``: clips whose normalized text occurs more than once
   (``duplicates.csv``, ``units.csv``);
3. ``plot_overlaps.py``: plots in ``plots/``.

``report.md`` sums them up: what was checked, which dev and test clips have a
text that is also in train and where it comes from, repeats inside and
between datasets, audio without metadata, and the plots. Everything goes to
``output/leak-clean/``. Paths come from the config (``config.yaml`` or
``ASR_CONFIG``).

    python datasets/leak-clean/check_leaks.py
    python datasets/leak-clean/check_leaks.py --datasets common_voice cv10_test
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from collect_metadata import collect
from find_overlaps import KINDS, find
from leak_common import CONFIG_PATH, DATASET_ORDER, ROOT, output_dir, unit_key
from plot_overlaps import LONG_TEXT_WORDS, PROBLEM_LABELS, SKIPPED_LABELS, plot

REPORT_NAME = "report.md"
EXAMPLES = 15
TEXT_WIDTH = 90

KIND_LABELS = {
    "between_datasets": "між різними датасетами",
    "between_splits": "між сплітами одного датасету",
    "within_split": "всередині одного спліту",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Datasets to check. Default: all.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: output/leak-clean.")
    return parser.parse_args()


def spaced(number: int) -> str:
    return f"{int(number):,}".replace(",", " ")


def share(part: int, total: int) -> str:
    return f"{100 * part / total:.1f} %" if total else "—"


def cell(text) -> str:
    """Text for one Markdown table cell: one line, no pipes, not too long."""
    value = " ".join(str(text).split()).replace("|", "\\|")
    return value if len(value) <= TEXT_WIDTH else value[: TEXT_WIDTH - 1] + "…"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return lines


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except OSError:
        return "—"
    return completed.stdout.strip() or "—"


def where(dataset: str, split: str) -> str:
    return f"{dataset}/{split}" if split else dataset


def eval_leaks(duplicates: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """One row per dev or test split: clips whose text is in some train, and where from."""
    totals = dict(zip(units["unit"], units["clips"]))
    evaluated = [unit for unit in units["unit"] if not unit.endswith("/train")]
    copies = duplicates[duplicates["split"] != "train"].copy()
    copies["unit"] = copies["dataset"] + "/" + copies["split"]
    copies["trains"] = copies["units"].map(lambda listed: [unit for unit in listed.split("; ") if unit.endswith("/train")])
    copies = copies[copies["trains"].map(bool)]
    rows = []
    for unit in evaluated:
        chosen = copies[copies["unit"] == unit]
        long = chosen[chosen["words"].astype(int) >= LONG_TEXT_WORDS]
        sources = Counter(train for trains in long["trains"] for train in trains)
        rows.append(
            {
                "unit": unit,
                "clips": int(totals[unit]),
                "every": len(chosen),
                "long": len(long),
                "sources": "; ".join(
                    f"{train} {spaced(count)}" for train, count in sorted(sources.items(), key=lambda item: -item[1])
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=["unit", "clips", "every", "long", "sources"])
    frame["long_share"] = frame["long"] / frame["clips"].where(frame["clips"] > 0)
    return frame


def write_report(
    folder: Path,
    datasets: list[str],
    metadata: pd.DataFrame,
    coverage: pd.DataFrame,
    duplicates: pd.DataFrame,
    units: pd.DataFrame,
    plots: list[Path],
) -> Path:
    coverage = coverage.assign(count=coverage["count"].astype(int))
    skipped = coverage[coverage["problem"].isin(SKIPPED_LABELS)]
    problems = coverage[coverage["problem"].isin(PROBLEM_LABELS) & (coverage["count"] > 0)]
    leaks = eval_leaks(duplicates, units)
    checked = sorted({unit.split("/")[0] for unit in units["unit"]}, key=lambda name: unit_key(name + "/"))
    missing = [name for name in datasets if name not in checked]
    groups = duplicates.drop_duplicates("group")

    lines = [
        "# Перевірка датасетів на повтори й витоки",
        "",
        f"Створено {datetime.now():%Y-%m-%d %H:%M}, коміт `{git_commit()}`, конфіг `{CONFIG_PATH.name}`. "
        f"Скрипт: `datasets/leak-clean/check_leaks.py`. "
        f"Тексти порівнюються після нормалізації: регістр, пунктуація й варіанти апострофа не враховуються. "
        f"Короткими вважаються тексти до {LONG_TEXT_WORDS - 1} слів: такі фрази збігаються майже неминуче.",
        "",
        "## Головне",
        "",
    ]
    not_checked_audio = sorted(
        {row.dataset for row in skipped.itertuples(index=False) if row.problem == "audio_not_checked"}
        | set(missing),
        key=lambda name: unit_key(name + "/"),
    )
    summary = [
        f"Перевірено {spaced(units['clips'].sum())} кліпів train, dev і test з текстом. "
        f"Датасети: {', '.join(checked) or '—'}.",
    ]
    if missing:
        summary.append(f"Не знайдено на диску, тому не перевірено: {', '.join(missing)}.")
    if not leaks.empty:
        worst = leaks.sort_values("long_share", ascending=False).iloc[0]
        flagged = leaks[leaks["long"] > 0]
        if flagged.empty:
            summary.append(f"Жоден dev чи test не має текстів від {LONG_TEXT_WORDS} слів, що є в train.")
        else:
            summary.append(
                f"Найбільший витік тексту в оцінку: `{worst['unit']}`, {spaced(worst['long'])} з {spaced(worst['clips'])} "
                f"кліпів ({share(worst['long'], worst['clips'])}) мають текст від {LONG_TEXT_WORDS} слів, що є в train. "
                f"Таких dev/test сплітів {len(flagged)} з {len(leaks)}."
            )
    summary.append(
        "Повтори текстів: "
        + "; ".join(
            f"{KIND_LABELS[kind]} {spaced((groups['kind'] == kind).sum())}"
            for kind in KINDS
        )
        + "."
    )
    pairs = f"{spaced(problems['count'].sum())} випадків, див. розділ 4." if not problems.empty else "не знайдено."
    if not_checked_audio:
        pairs += f" Аудіо не перевірено для: {', '.join(not_checked_audio)}."
    summary.append("Аудіо й метадані без пари: " + pairs)
    lines += [f"{number}. {text}" for number, text in enumerate(summary, start=1)]

    lines += ["", "## 1. Що перевірено", ""]
    lines += table(
        ["Датасет/спліт", "Кліпів з текстом"],
        [[unit, spaced(clips)] for unit, clips in zip(units["unit"], units["clips"])],
    )
    if not skipped.empty:
        lines += ["", "Не перевірено:", ""]
        lines += [
            f"{number}. `{where(row.dataset, row.split)}`: {SKIPPED_LABELS[row.problem]} (`{row.example}`)."
            for number, row in enumerate(skipped.itertuples(index=False), start=1)
        ]

    lines += [
        "",
        "## 2. Тексти dev і test, що є в train",
        "",
        f"Частка кліпів оцінки, нормалізований текст яких трапляється в будь-якому train-спліті. "
        f"«Від {LONG_TEXT_WORDS} слів» — головна колонка: це речення, які модель могла запам'ятати. "
        f"«Звідки» показує, у яких train-сплітах є ці тексти від {LONG_TEXT_WORDS} слів.",
        "",
    ]
    if leaks.empty:
        lines.append("Немає жодного dev чи test спліту.")
    else:
        lines += table(
            ["Спліт", "Кліпів", "Усі тексти", f"Від {LONG_TEXT_WORDS} слів", "Звідки"],
            [
                [
                    row.unit,
                    spaced(row.clips),
                    f"{spaced(row.every)} ({share(row.every, row.clips)})",
                    f"{spaced(row.long)} ({share(row.long, row.clips)})",
                    row.sources or "—",
                ]
                for row in leaks.itertuples(index=False)
            ],
        )
        if leaks["unit"].str.startswith("common_voice/").any():
            lines += [
                "",
                "Для `common_voice` це очікувано: `train_validated.tsv` свідомо зберігає речення й дикторів, "
                "спільних із dev і test, а без них train був би майже вдвічі меншим. Що це не завищує WER, перевірено "
                "скриптом `score_common_voice_cleaned.py` (див. `datasets/common-voice/README.md`).",
            ]
        leaked = duplicates[
            (duplicates["split"] != "train")
            & duplicates["units"].str.contains("/train", regex=False)
            & (duplicates["words"].astype(int) >= LONG_TEXT_WORDS)
        ].drop_duplicates("group")
        if not leaked.empty:
            lines += ["", f"Приклади (найчастіші тексти, до {EXAMPLES}):", ""]
            lines += table(
                ["Текст", "Копій", "Де трапляється"],
                [[row.text_norm, row.copies, row.units] for row in leaked.sort_values("copies", ascending=False).head(EXAMPLES).itertuples(index=False)],
            )

    lines += [
        "",
        "## 3. Повтори текстів",
        "",
        "Група — один нормалізований текст, що трапляється в кількох кліпах. Повтор усередині спліту не є витоком, "
        "але дає цьому тексту більшу вагу в навчанні чи оцінці.",
        "",
    ]
    lines += table(
        ["Тип", "Текстів", "Кліпів", f"Текстів від {LONG_TEXT_WORDS} слів"],
        [
            [
                KIND_LABELS[kind],
                spaced((groups["kind"] == kind).sum()),
                spaced(groups.loc[groups["kind"] == kind, "copies"].sum()),
                spaced(((groups["kind"] == kind) & (groups["words"].astype(int) >= LONG_TEXT_WORDS)).sum()),
            ]
            for kind in KINDS
        ],
    )
    between = groups[(groups["kind"] == "between_datasets") & (groups["words"].astype(int) >= LONG_TEXT_WORDS)]
    if not between.empty:
        lines += ["", f"Тексти від {LONG_TEXT_WORDS} слів, спільні для різних датасетів (до {EXAMPLES}):", ""]
        lines += table(
            ["Текст", "Копій", "Де трапляється"],
            [[row.text_norm, row.copies, row.units] for row in between.sort_values("copies", ascending=False).head(EXAMPLES).itertuples(index=False)],
        )

    lines += ["", "## 4. Аудіо й метадані без пари", ""]
    if problems.empty:
        lines.append("До кожного перевіреного аудіо є метадані, і до кожного рядка метаданих є аудіо.")
        if not_checked_audio:
            lines += ["", f"Аудіо не перевірено для: {', '.join(not_checked_audio)} (див. розділ 1)."]
    else:
        lines += table(
            ["Датасет/спліт", "Проблема", "Кількість", "Приклад"],
            [
                [where(row.dataset, row.split), PROBLEM_LABELS[row.problem], spaced(row.count), row.example]
                for row in problems.itertuples(index=False)
            ],
        )
        lines += [
            "",
            "Для Common Voice «аудіо без метаданих» — це, зокрема, кліпи, прибрані з train скриптом "
            "`clean_common_voice_train.py`, і кліпи з `invalidated`, якщо вони лежать у `clips/`.",
        ]

    lines += ["", "## 5. Графіки", ""]
    for path in plots:
        lines += [f"![{path.stem}]({path.relative_to(folder).as_posix()})", ""]

    lines += [
        "## 6. Файли",
        "",
        "1. `metadata.parquet` — один рядок на кліп: датасет, спліт, ID, ім'я аудіо, точне місце, текст.",
        "2. `coverage.csv` — усі перевірки аудіо й метаданих, включно з тими, що нічого не знайшли.",
        "3. `duplicates.csv` — кожен кліп, текст якого трапляється більше одного разу, з групою і всіма сплітами цього тексту.",
        "4. `units.csv` — кількість кліпів у кожному спліті.",
        "",
    ]
    path = folder / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    folder = output_dir(args.output)
    print("== 1/3 collect metadata", flush=True)
    metadata, coverage = collect(args.datasets, folder)
    if metadata.empty:
        raise SystemExit(f"No clips with text found. See {folder / 'coverage.csv'}.")
    print("== 2/3 find overlaps", flush=True)
    duplicates, units = find(folder)
    print("== 3/3 plot", flush=True)
    plots = plot(folder)
    report = write_report(folder, args.datasets, metadata, coverage, duplicates, units, plots)
    print(f"report: {report}", flush=True)


if __name__ == "__main__":
    main()
