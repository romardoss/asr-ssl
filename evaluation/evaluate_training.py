"""Evaluate a training run: its first-epoch checkpoint and its final one.

The run is a checkpoint directory, or the newest run of a stage:
``fine-tune`` (``model.name``) or ``student`` (``student.name``). Two
checkpoints are scored with ``evaluate.py``, which adds each to
``<paths.evaluation>/results.csv``:

1. ``<name>-epoch00.ckpt``, saved at the end of the first epoch;
2. the checkpoint ``last.ckpt`` points to, the final one of a finished run.

With one epoch they are the same file and it is scored once. ``--final-only``
scores only the second one. A checkpoint
that already has a row in ``results.csv`` is skipped, so running this again
after a crash scores only what is missing.

    python evaluation/evaluate_training.py --stage fine-tune
    python evaluation/evaluate_training.py --stage student
    python evaluation/evaluate_training.py --checkpoint-dir /path/to/model-whisper-uk_02
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINE_TUNE = ROOT / "fine-tune"
EVALUATE = ROOT / "evaluation" / "evaluate.py"
for folder in (FINE_TUNE, ROOT / "evaluation"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import results_table  # noqa: E402
from checkpoints import latest_checkpoint, latest_student_checkpoint, numbered_checkpoint_dir  # noqa: E402
from settings import load_settings, load_student_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--stage", choices=("fine-tune", "student"), help="Score the newest run of this stage.")
    target.add_argument("--checkpoint-dir", type=Path, help="Score this checkpoint directory.")
    parser.add_argument("--final-only", action="store_true", help="Score only the final checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be scored.")
    return parser.parse_args()


def stage_dir(stage: str) -> Path:
    if stage == "fine-tune":
        cfg = load_settings()
        return numbered_checkpoint_dir(cfg.checkpoints, cfg.model_name, resume=True)
    cfg = load_student_settings()
    found = latest_student_checkpoint(cfg.checkpoints, cfg.name)
    if found is None:
        raise SystemExit(f"No student checkpoint for {cfg.name} under {cfg.checkpoints}")
    return found.parent


def targets(directory: Path, final_only: bool = False) -> list[Path]:
    """The first-epoch checkpoint and the final one, without repeats."""
    chosen: list[Path] = []
    if not final_only:
        first = sorted(path for path in directory.glob("*-epoch00.ckpt") if path.is_file())
        chosen.extend(first[:1])
    final = latest_checkpoint(directory)
    if final is not None:
        chosen.append(final)
    unique: dict[Path, Path] = {}
    for path in chosen:
        unique.setdefault(path.resolve(), path.resolve())
    return list(unique.values())


def main() -> None:
    args = parse_args()
    directory = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else stage_dir(args.stage)
    if not directory.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {directory}")
    checkpoints = targets(directory, args.final_only)
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {directory}")

    done = results_table.evaluated_checkpoints(load_settings().evaluation_dir)
    for checkpoint in checkpoints:
        if str(checkpoint) in done:
            print(f"skip {checkpoint}: already in results.csv", flush=True)
            continue
        command = [sys.executable, str(EVALUATE), "--model", str(checkpoint)]
        print("+ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"evaluate.py exited with status {completed.returncode} for {checkpoint}")


if __name__ == "__main__":
    main()
