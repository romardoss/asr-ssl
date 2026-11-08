"""Mark the original labels of the labelled datasets with the judge and report per dataset/split.

The judging is the ``Judge`` of ``pseudo-labelling/llm-as-a-judge/judge.py``:
same ``prompt.txt``, 0-5 marks, retries, and ``OPENROUTER_API_KEY`` from the
environment or ``.env``. Two backends (see ``decisions.py``):

1. ``llm``: a chat model, ``--model``, else ``OPENROUTER_MODEL``, else ``pipeline.judge_model``;
2. ``decisions``: a decision model, ``--model``, else ``OPENROUTER_DECISION_MODEL``.
   ``marks.csv`` then also holds the probability-weighted ``score`` and ``confidence``.

``USE_DECISION_INSTEAD_OF_LLM`` picks the backend; ``--backend`` overrides it.

Every split of every dataset in ``fine-tune/dataset_roles.py`` that exists on
disk is marked in full. ``--per-split N`` marks a random sample of N
transcripts per dataset/split instead (fixed ``--seed``).
Only transcripts are read, audio is not decoded.

Each run gets its own folder, so another judge never overwrites a report:

    analysis/label-judge/<model>-NN/
        sample.csv                  the transcripts sent to the judge
        marks.csv                   index,ID,mark[,score,confidence] (appended as batches finish)
        report-<model>.md           tables, plots, lowest-marked examples
        plots/*.png

A stopped run continues with ``--resume <folder>``. ``--report-only <folder>``
rebuilds the report from the saved marks without calling the judge.

    python analysis/label-judge/judge_labels.py
    python analysis/label-judge/judge_labels.py --per-split 500
    python analysis/label-judge/judge_labels.py --model google/gemini-2.5-pro --datasets ukr_dialects fleurs
    python analysis/label-judge/judge_labels.py --backend decisions --workers 16
    python analysis/label-judge/judge_labels.py --resume analysis/label-judge/openai_gpt-4.1-01
    python analysis/label-judge/judge_labels.py --report-only analysis/label-judge/openai_gpt-4.1-01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINE_TUNE = ROOT / "fine-tune"
JUDGE_PATH = ROOT / "pseudo-labelling" / "llm-as-a-judge" / "judge.py"
for path in (ROOT, FINE_TUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import decisions  # noqa: E402
from dataset_roles import ROLES  # noqa: E402
from project_config import config_int, load_config  # noqa: E402


def _load_judge():
    spec = importlib.util.spec_from_file_location("llm_judge", JUDGE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {JUDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge_module = _load_judge()
CONFIG = load_config()
SPLITS = ("train", "dev", "test")
MARKS = tuple(range(6))
# Same mark colors as pseudo-labelling/llm-as-a-judge/plot_marks.py: red (0) to green (5).
COLORS = ("#9b2226", "#bb3e03", "#ca6702", "#ee9b00", "#90a955", "#4d7c0f")
SAMPLE_FIELDS = ["index", "dataset", "split", "ID", "text"]

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend",
        choices=decisions.BACKENDS,
        default=None,
        help="llm or decisions. Default: decisions when USE_DECISION_INSTEAD_OF_LLM is true, else llm.",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model id of the judge.")
    parser.add_argument("--datasets", nargs="+", default=list(ROLES), choices=list(ROLES))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--per-split", type=int, default=0, help="Random transcripts per dataset/split; 0 (default) marks all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=config_int(CONFIG, "pipeline.judge_workers"))
    parser.add_argument("--retries", type=int, default=config_int(CONFIG, "pipeline.judge_retries"))
    parser.add_argument("--out", type=Path, default=HERE, help="Folder that holds the run folders.")
    parser.add_argument("--resume", type=Path, default=None, help="Continue the run in this folder.")
    parser.add_argument("--report-only", type=Path, default=None, help="Rebuild the report of this run folder.")
    args = parser.parse_args()
    if args.per_split < 0:
        parser.error("--per-split must be 0 or more")
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be at least 1")
    return args


def judge_model(backend: str, cli: str | None) -> str:
    if cli:
        return cli.strip()
    env = decisions.model_for(backend)
    if env or backend == "decisions":
        return env
    configured = (CONFIG.get("pipeline") or {}).get("judge_model") or ""
    return str(configured).strip()


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "judge"


def new_run_dir(out: Path, slug: str) -> Path:
    for number in range(1, 1000):
        path = out / f"{slug}-{number:02d}"
        if not path.exists():
            return path
    raise SystemExit(f"No free run folder for {slug} in {out}")


# ---------------------------------------------------------------- data


def part_rows(part) -> list[tuple[str, str]]:
    """(example id, transcript) of one dataset part, without touching the audio."""
    if hasattr(part, "sentences"):
        return [(part.example_id(i), str(text)) for i, text in enumerate(part.sentences)]
    if hasattr(part, "records"):
        return [(record[3], record[2]) for record in part.records]
    return [(row[4], row[3]) for row in part.rows]


def build_sample(datasets: list[str], splits: list[str], per_split: int, seed: int) -> tuple[list[dict], dict]:
    from settings import load_settings
    from speech_data import _part

    settings = load_settings()
    rows: list[dict] = []
    population: dict[str, int] = {}
    for name in datasets:
        for split in splits:
            if split not in ROLES[name]:
                continue
            try:
                items = part_rows(_part(settings, name, split))
            except SystemExit as error:
                print(f"skip {name}/{split}: {error}", file=sys.stderr, flush=True)
                continue
            items = [(item_id, " ".join(text.split())) for item_id, text in items if text.strip()]
            population[f"{name}/{split}"] = len(items)
            chosen = items
            if per_split and len(items) > per_split:
                picked = sorted(random.Random(f"{seed}:{name}:{split}").sample(range(len(items)), per_split))
                chosen = [items[i] for i in picked]
            print(f"{name}/{split}: {len(chosen)} of {len(items)}", flush=True)
            for item_id, text in chosen:
                rows.append({"index": len(rows), "dataset": name, "split": split, "ID": item_id, "text": text})
    return rows, population


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- judging


def run_judge(run_dir: Path, rows: list[dict], backend: str, model: str, workers: int, retries: int) -> None:
    judge = judge_module.Judge(backend, model)
    items = [(int(row["index"]), row["ID"], row["text"]) for row in rows]
    failed = judge.mark_rows(items, run_dir / "marks.csv", workers, retries)
    if failed:
        print(f"{len(failed)} transcripts left unmarked; rerun with --resume {run_dir}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- report


def group_marks(run_dir: Path) -> tuple[dict, dict, list[str], dict]:
    sample = read_csv(run_dir / "sample.csv")
    marks: dict[int, int] = {}
    extras: dict[int, tuple[float, float]] = {}
    marks_path = run_dir / "marks.csv"
    if marks_path.is_file():
        for row in judge_module.valid_mark_rows(marks_path):
            marks[int(row["index"])] = int(row["mark"])
            try:
                extras[int(row["index"])] = (float(row["score"]), float(row["confidence"]))
            except (KeyError, TypeError, ValueError):
                pass
    order: list[str] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    decided: dict[str, list[tuple[float, float]]] = defaultdict(list)
    examples: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in sample:
        group = f"{row['dataset']}/{row['split']}"
        if group not in order:
            order.append(group)
        mark = marks.get(int(row["index"]))
        if mark is not None:
            by_group[group].append(mark)
            examples[group].append((mark, row["text"]))
        if int(row["index"]) in extras:
            decided[group].append(extras[int(row["index"])])
    return by_group, examples, order, decided


def summary(values: list[int]) -> dict:
    count = len(values)
    if not count:
        return {"n": 0}
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / (count - 1) if count > 1 else 0.0
    counts = Counter(values)
    return {
        "n": count,
        "mean": mean,
        "ci": 1.96 * math.sqrt(variance / count),
        "shares": [counts[mark] / count for mark in MARKS],
        "good": sum(counts[m] for m in (4, 5)) / count,
        "bad": sum(counts[m] for m in (0, 1, 2)) / count,
    }


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_distribution(groups: list[str], stats: dict, model: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 0.55 * len(groups) + 2.2))
    positions = list(range(len(groups)))[::-1]
    left = [0.0] * len(groups)
    for mark in MARKS:
        widths = [stats[g]["shares"][mark] * 100 for g in groups]
        axis.barh(positions, widths, left=left, color=COLORS[mark], edgecolor="white", linewidth=1.5, height=0.7, label=str(mark))
        for y, x0, width in zip(positions, left, widths):
            if width >= 6:
                axis.text(x0 + width / 2, y, f"{width:.0f}", ha="center", va="center", fontsize=9, color="white")
        left = [a + b for a, b in zip(left, widths)]
    axis.set_yticks(positions, [f"{g}  (n={stats[g]['n']})" for g in groups])
    axis.set_xlim(0, 100)
    axis.set_xlabel("share of transcripts (%)")
    axis.set_title(f"Judge marks of original labels\n{model}")
    axis.grid(axis="y", visible=False)
    axis.legend(title="mark", ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / max(len(groups), 1)), frameon=False)
    save(figure, path)


def plot_mean(groups: list[str], stats: dict, model: str, path: Path) -> None:
    ranked = sorted(groups, key=lambda g: stats[g]["mean"])
    figure, axis = plt.subplots(figsize=(9, 0.45 * len(groups) + 2))
    positions = range(len(ranked))
    means = [stats[g]["mean"] for g in ranked]
    axis.errorbar(means, positions, xerr=[stats[g]["ci"] for g in ranked], fmt="o", color="#1d4e89", ecolor="#7a8fa6", capsize=3, markersize=7)
    for y, mean in zip(positions, means):
        axis.annotate(f"{mean:.2f}", xy=(mean, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    axis.set_yticks(list(positions), ranked)
    axis.set_xlim(0, 5)
    axis.set_xlabel("mean mark (95% CI)")
    axis.set_title(f"Mean judge mark by dataset/split\n{model}")
    save(figure, path)


def plot_bad_share(groups: list[str], stats: dict, model: str, path: Path) -> None:
    ranked = sorted(groups, key=lambda g: -stats[g]["bad"])
    figure, axis = plt.subplots(figsize=(9, 0.45 * len(groups) + 2))
    values = [stats[g]["bad"] * 100 for g in ranked][::-1]
    bars = axis.barh(range(len(ranked)), values, color=COLORS[1], height=0.6)
    for bar, value in zip(bars, values):
        axis.annotate(f"{value:.1f}%", xy=(value, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    axis.set_yticks(range(len(ranked)), ranked[::-1])
    axis.set_xlim(0, max(max(values, default=0) * 1.2, 5))
    axis.set_xlabel("share of transcripts marked 0-2 (%)")
    axis.set_title(f"Labels the judge finds poor\n{model}")
    save(figure, path)


def write_report(run_dir: Path) -> Path:
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model, slug = info["model"], model_slug(info["model"])
    by_group, examples, order, decided = group_marks(run_dir)
    groups = [g for g in order if by_group.get(g)]
    if not groups:
        raise SystemExit(f"No marks in {run_dir} yet.")
    stats = {g: summary(by_group[g]) for g in groups}
    plots = run_dir / "plots"
    plot_distribution(groups, stats, model, plots / "mark_distribution.png")
    plot_mean(groups, stats, model, plots / "mean_mark.png")
    plot_bad_share(groups, stats, model, plots / "poor_share.png")

    sampled = sum(1 for _ in read_csv(run_dir / "sample.csv"))
    judged = sum(s["n"] for s in stats.values())
    lines = [
        f"# Judge marks of original labels — {model}",
        "",
        f"- judge model: `{model}` (backend `{info.get('backend', 'llm')}`)",
        f"- started: {info['started']}, report: {datetime.now().isoformat(timespec='seconds')}",
        f"- prompt: `pseudo-labelling/llm-as-a-judge/prompt.txt` (sha256 `{info['prompt_sha256'][:12]}`)",
        f"- sample: " + (f"up to {info['per_split']} transcripts per dataset/split, seed {info['seed']}" if info["per_split"] else "all transcripts"),
        f"- marked: {judged} of {sampled} sampled transcripts",
        "",
        "Marks: 0 nonsense · 1 mostly broken · 2 many errors · 3 several errors · 4 minor errors · 5 sound label.",
        "",
        *(["Decision backend: the mark is the most probable level; *score* is the probability-weighted level "
           "(0-5) and *confidence* is how concentrated the probabilities are.", ""] if decided else []),
        "## Summary",
        "",
        "| dataset/split | population | marked | mean ± 95% CI | " + " | ".join(f"% {m}" for m in MARKS) + " | % 4-5 | % 0-2 |"
        + (" mean score | mean confidence |" if decided else ""),
        "|---|---:|---:|---:|" + "---:|" * (len(MARKS) + 2 + (2 if decided else 0)),
    ]
    for g in groups:
        s = stats[g]
        shares = " | ".join(f"{share * 100:.1f}" for share in s["shares"])
        population = info["population"].get(g, "")
        row = f"| {g} | {population} | {s['n']} | {s['mean']:.2f} ± {s['ci']:.2f} | {shares} | {s['good'] * 100:.1f} | {s['bad'] * 100:.1f} |"
        if decided:
            pairs = decided.get(g) or []
            if pairs:
                row += f" {sum(p[0] for p in pairs) / len(pairs):.2f} | {sum(p[1] for p in pairs) / len(pairs):.2f} |"
            else:
                row += " | |"
        lines.append(row)
    lines += [
        "",
        "## Plots",
        "",
        "![Mark distribution](plots/mark_distribution.png)",
        "",
        "![Mean mark](plots/mean_mark.png)",
        "",
        "![Poor labels](plots/poor_share.png)",
        "",
        "## Lowest-marked examples",
        "",
    ]
    for g in groups:
        worst = sorted(examples[g], key=lambda pair: pair[0])[:5]
        lines.append(f"### {g}")
        lines.append("")
        lines += [f"- **{mark}** — {text}" for mark, text in worst]
        lines.append("")
    path = run_dir / f"report-{slug}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    if args.report_only:
        print(f"wrote {write_report(args.report_only)}")
        return

    if args.resume:
        run_dir = args.resume
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        backend, model = info.get("backend", "llm"), info["model"]
    else:
        backend = args.backend or decisions.configured_backend()
        model = judge_model(backend, args.model)
        if not model:
            source = decisions.MODEL_ENV if backend == "decisions" else f"{decisions.LLM_MODEL_ENV} / pipeline.judge_model"
            raise SystemExit(f"No {backend} judge model: pass --model or set {source}.")
        # Fail on a missing key or prompt before reading every dataset.
        judge_module.Judge(backend, model)
        rows, population = build_sample(args.datasets, args.splits, args.per_split, args.seed)
        if not rows:
            raise SystemExit("No transcripts found for the selected datasets/splits.")
        run_dir = new_run_dir(args.out, model_slug(model))
        run_dir.mkdir(parents=True)
        write_csv(run_dir / "sample.csv", SAMPLE_FIELDS, rows)
        info = {
            "model": model,
            "backend": backend,
            "started": datetime.now().isoformat(timespec="seconds"),
            "datasets": args.datasets,
            "splits": args.splits,
            "per_split": args.per_split,
            "seed": args.seed,
            "population": population,
            "prompt_sha256": hashlib.sha256(judge_module.PROMPT_PATH.read_bytes()).hexdigest(),
        }
        (run_dir / "run.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"run folder {run_dir}", flush=True)

    run_judge(run_dir, read_csv(run_dir / "sample.csv"), backend, model, args.workers, args.retries)
    print(f"wrote {write_report(run_dir)}")


if __name__ == "__main__":
    main()
