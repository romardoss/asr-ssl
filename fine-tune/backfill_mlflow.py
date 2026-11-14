"""Load finished runs into MLflow, so old experiments show up next to new ones.

Two kinds of folders are read:

- ``<paths.fine_tune_output>/<model>/version_N`` (``metrics.csv``, ``meta.yaml``,
  ``config.yaml``): a training run, every metric by step and by epoch;
- ``<paths.evaluation>/<model>-XX`` (``results.json``, ``meta.yaml``): an evaluation run.

A folder that is already in MLflow (tag ``source_dir``) is skipped, so the script can
run again at any time. It writes into the store of the ``mlflow`` config section.

    python fine-tune/backfill_mlflow.py
    python fine-tune/backfill_mlflow.py --only evaluation
    python fine-tune/backfill_mlflow.py --training-dir /other/output/fine-tune --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

import tracking  # noqa: E402
from project_config import load_config, repo_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("training", "evaluation"), help="Load only one kind of run.")
    parser.add_argument("--training-dir", type=Path, help="Default: paths.fine_tune_output.")
    parser.add_argument("--evaluation-dir", type=Path, help="Default: paths.evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="List the folders and write nothing.")
    return parser.parse_args()


def training_versions(root: Path) -> list[tuple[str, Path]]:
    found = []
    for model in sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []:
        for version in sorted(model.glob("version_*"), key=lambda path: int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else 0):
            if (version / "metrics.csv").is_file():
                found.append((model.name, version))
    return found


def evaluation_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/results.json")) if root.is_dir() else []


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    if not tracking.load_settings(config).enabled:
        raise SystemExit("MLflow is off: add the mlflow section (enabled: true) to the config.")
    paths = config.get("paths") or {}
    training_dir = args.training_dir or repo_path(paths.get("fine_tune_output", "output/fine-tune"))
    evaluation_dir = args.evaluation_dir or repo_path(paths.get("evaluation", "evaluation"))

    trainings = [] if args.only == "evaluation" else training_versions(training_dir)
    evaluations = [] if args.only == "training" else evaluation_runs(evaluation_dir)
    print(f"training runs: {len(trainings)} in {training_dir}")
    print(f"evaluation runs: {len(evaluations)} in {evaluation_dir}")
    if args.dry_run:
        for model, version in trainings:
            print(f"  train {model}/{version.name}")
        for run in evaluations:
            print(f"  eval  {run.name}")
        return

    saved = failed = 0
    for model, version in trainings:
        if tracking.log_training_version(version, model):
            saved += 1
        else:
            failed += 1
            print(f"not loaded: {version}")
    for run in evaluations:
        if tracking.log_evaluation(run):
            saved += 1
        else:
            failed += 1
            print(f"not loaded: {run}")
    print(f"in MLflow: {saved}, not loaded: {failed}")


if __name__ == "__main__":
    main()
