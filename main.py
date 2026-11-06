"""Start a project stage. Settings come from ``config.yaml``, or from the
file named by ``ASR_CONFIG`` (for example ``config.vast.yaml``).

    ASR_CONFIG=config.vast.yaml python main.py pipeline

    python main.py pipeline
        Run the stages in ``pipeline`` in order: language filter, 30-second
        splits, teacher labelling, judge marks, student training, evaluation.
        Each stage is run, skip, or skip_if_present.
    python main.py pipeline --dry-run
        Print those stages and the commands they would start.
    python main.py fine-tune
        Fine-tune the Whisper decoder on ``data.train``. The encoder stays
        frozen. Dev is scored during training; each ``data.test`` set is
        scored after it.
    python main.py student
        Fine-tune a student decoder on VOA slices and their pseudo-labels.
        Dev and the final test are Common Voice.
    python main.py evaluate
        Score ``model.id`` on each configured test set and on all of them
        combined. Writes results under ``paths.evaluation``.
    python main.py evaluate --model /path/to/last.ckpt
        Same scoring, loading that checkpoint on top of ``model.id``.
    python main.py label
        Transcribe Ukrainian VOA slices with ``pseudo_labelling.teacher_model``.
        Appends ``labels.csv`` under ``pseudo_labelling.output_dir``.
    python main.py judge --max-examples 32
        Ask the OpenRouter judge to mark pseudo-label quality and append
        ``label_marks.csv``. ``--max-examples`` limits how many rows are sent.
    python main.py plot --watch
        Plot loss and error rates from a fine-tune or student ``metrics.csv``.
        ``--watch`` redraws the plots while that file changes.
    python main.py marks
        Bar plot of the judge marks in ``label_marks.csv``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(script: Path, extra: list[str]) -> None:
    if not script.is_file():
        raise SystemExit(f"Script not found: {script}")
    completed = subprocess.run([sys.executable, str(script), *extra], cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _present(args: argparse.Namespace, *names: str) -> list[str]:
    extra: list[str] = []
    for name in names:
        value = getattr(args, name)
        flag = "--" + name.replace("_", "-")
        if value is True:
            extra.append(flag)
        elif value is not None and value is not False:
            extra.extend([flag, str(value)])
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    pipeline = commands.add_parser("pipeline", help="Run the configured pipeline stages.")
    pipeline.add_argument("--dry-run", action="store_true", help="Print each stage without starting it.")

    commands.add_parser("fine-tune", help="Fine-tune the Whisper decoder on data.train.")
    commands.add_parser("student", help="Fine-tune the student on VOA pseudo-labels.")

    evaluate = commands.add_parser("evaluate", help="Score each test set and the combined test set.")
    evaluate.add_argument(
        "--model",
        default=None,
        help="Checkpoint (.ckpt) or model directory. Default: model.id from config.yaml.",
    )

    commands.add_parser("label", help="Pseudo-label VOA slices with the teacher model.")

    judge = commands.add_parser("judge", help="Score pseudo-labels with the OpenRouter judge.")
    judge.add_argument("--labels", type=Path, default=None, help="Pseudo-label CSV to score.")
    judge.add_argument("--max-examples", type=int, default=None, help="Score only the first N rows.")
    judge.add_argument("--workers", type=int, default=None, help="How many requests may be in flight.")
    judge.add_argument("--retries", type=int, default=None, help="Attempts per batch.")

    plot = commands.add_parser("plot", help="Plot fine-tune or student metrics.")
    plot.add_argument(
        "--model-name", default=None, help="Directory name under paths.fine_tune_output. Default: model.name."
    )
    plot.add_argument("--version", default=None, help="Version number or folder. Default: newest.")
    plot.add_argument("--watch", action="store_true", help="Refresh while metrics.csv changes.")
    plot.add_argument("--interval", type=float, default=None, help="Seconds between checks in --watch mode.")

    marks = commands.add_parser("marks", help="Plot the judge mark distribution.")
    marks.add_argument("--marks", type=Path, default=None, help="label_marks.csv to summarise.")

    args = parser.parse_args()
    scripts = {
        "pipeline": ROOT / "pipeline.py",
        "fine-tune": ROOT / "fine-tune" / "fine-tune.py",
        "student": ROOT / "fine-tune" / "fine-tune-student.py",
        "evaluate": ROOT / "evaluation" / "evaluate.py",
        "label": ROOT / "pseudo-labelling" / "label.py",
        "judge": ROOT / "pseudo-labelling" / "llm-as-a-judge" / "judge.py",
        "plot": ROOT / "fine-tune" / "plot_metrics.py",
        "marks": ROOT / "pseudo-labelling" / "llm-as-a-judge" / "plot_marks.py",
    }
    forwarded = {
        "pipeline": ("dry_run",),
        "fine-tune": (),
        "student": (),
        "evaluate": ("model",),
        "label": (),
        "judge": ("labels", "max_examples", "workers", "retries"),
        "plot": ("model_name", "version", "watch", "interval"),
        "marks": ("marks",),
    }
    _run(scripts[args.command], _present(args, *forwarded[args.command]))


if __name__ == "__main__":
    main()
