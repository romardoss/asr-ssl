"""Keep ``<paths.evaluation>/results.csv``: one row for each evaluation run.

``evaluate.py`` adds the row of its run when it finishes. A run already in the
table is replaced, so scoring the same folder again never duplicates it. The
files of each run (``results.json``, ``meta.yaml``, ``config.yaml``) stay the
source of truth. Running this script rebuilds the whole table from them.

Error rates are in percent. ``combined_weighted`` pools every clip.
``combined_mean`` is the plain mean over test sets.

    python evaluation/results_table.py
    python evaluation/results_table.py --evaluation evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import load_config, repo_path  # noqa: E402

TABLE_NAME = "results.csv"
RESULTS_NAME = "results.json"
META_NAME = "meta.yaml"
CONFIG_NAME = "config.yaml"

RUN_COLUMNS = (
    "run",
    "evaluated_at",
    "note",
    "model_type",
    "model",
    "checkpoint",
    "epoch",
    "batch_percent",
    "checkpoint_path",
    "init",
    "train_data",
    "accepted_marks",
    "cv_train_rows",
    "learning_rate",
    "max_epochs",
    "precision",
    "test_clips",
    "loss",
)
# Known test sets come first, in this order. Any other set follows by name.
TEST_SET_ORDER = ("combined_weighted", "combined_mean", "common_voice", "fleurs", "ukr_dialects", "cv10_test")
RATES = ("wer", "cer")
# Older results.json files call the pooled score "combined".
LEGACY_NAMES = {"combined": "combined_weighted"}
_LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _join(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _log_time(run: Path) -> str | None:
    log = run / "test.log"
    if not log.is_file():
        return None
    with log.open(encoding="utf-8", errors="replace") as handle:
        match = _LOG_TIME.match(handle.readline())
    return match.group(1) if match else None


def _model_type(kind: str | None, model: str | None, meta: dict, config: dict) -> str:
    if kind == "hf":
        return "pretrained"
    student = config.get("student") if isinstance(config.get("student"), dict) else {}
    if "init" in meta or (model is not None and model == student.get("name")):
        return "student"
    return "fine-tune"


def run_row(run: Path, evaluation_dir: Path) -> dict | None:
    """The table row of one run folder, or None when it has no results."""
    results = run / RESULTS_NAME
    if not results.is_file():
        return None
    payload = json.loads(results.read_text(encoding="utf-8"))
    meta = _load_yaml(run / META_NAME)
    config = _load_yaml(run / CONFIG_NAME)
    kind = payload.get("kind")
    model = meta.get("model") or payload.get("model")
    model_type = _model_type(kind, model, meta, config)

    # Training fields come from meta.yaml. Older runs only have them in config.yaml.
    section = config.get("student" if model_type == "student" else "training") or {}
    data = config.get("data") or {}
    if model_type == "student":
        accepted = section.get("accepted_mark") if section.get("filter_marks", True) else None
        train_data = ["voa_pseudo_labels"] if section else None
    else:
        accepted = None
        train_data = data.get("train")
    pretrained = model_type == "pretrained"

    row = {
        "run": run.relative_to(evaluation_dir).as_posix(),
        "evaluated_at": meta.get("evaluated_at") or _log_time(run),
        "note": meta.get("note"),
        "model_type": model_type,
        "model": model,
        "checkpoint": meta.get("checkpoint"),
        "epoch": meta.get("epoch"),
        "batch_percent": meta.get("batch_percent"),
        "checkpoint_path": None if pretrained else payload.get("model"),
        "init": meta.get("init") or (None if pretrained else section.get("init")),
        "train_data": None if pretrained else _join(meta.get("train_data") or train_data),
        "accepted_marks": _join(meta.get("accepted_mark") or accepted),
        "cv_train_rows": meta.get("cv_train_rows"),
        "learning_rate": None if pretrained else meta.get("learning_rate") or section.get("learning_rate"),
        "max_epochs": None if pretrained else meta.get("max_epochs") or section.get("max_epochs"),
        "precision": None if pretrained else section.get("precision"),
        "loss": payload.get("loss"),
    }
    test_sets = {LEGACY_NAMES.get(name, name): metrics for name, metrics in payload.get("test_sets", {}).items()}
    pooled = test_sets.get("combined_weighted", {})
    row["test_clips"] = int(pooled["clips"]) if "clips" in pooled else None
    for name, metrics in test_sets.items():
        for rate in RATES:
            if rate in metrics:
                row[f"{rate}_{name}"] = round(float(metrics[rate]) * 100, 4)
    return row


def _metric_columns(rows: list[dict]) -> list[str]:
    present = {key for row in rows for key in row if key not in RUN_COLUMNS}
    names = {key.split("_", 1)[1] for key in present}
    ordered = [name for name in TEST_SET_ORDER if name in names]
    ordered += sorted(names - set(ordered))
    return [f"{rate}_{name}" for rate in RATES for name in ordered if f"{rate}_{name}" in present]


def _write(path: Path, rows: list[dict]) -> None:
    columns = list(RUN_COLUMNS) + _metric_columns(rows)
    rows = sorted(rows, key=lambda row: (str(row.get("evaluated_at") or ""), row["run"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in columns})
    temporary.replace(path)


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: (value if value != "" else None) for key, value in row.items()} for row in csv.DictReader(handle)]


def add_run(run: Path, evaluation_dir: Path) -> Path:
    """Add or replace the row of ``run`` in ``<evaluation_dir>/results.csv``."""
    row = run_row(run, evaluation_dir)
    if row is None:
        raise SystemExit(f"No {RESULTS_NAME} in {run}")
    table = evaluation_dir / TABLE_NAME
    rows = [existing for existing in _read(table) if existing.get("run") != row["run"]]
    rows.append(row)
    _write(table, rows)
    return table


def evaluated_checkpoints(evaluation_dir: Path) -> set[str]:
    """Checkpoint paths that already have a row in the table."""
    return {row["checkpoint_path"] for row in _read(evaluation_dir / TABLE_NAME) if row.get("checkpoint_path")}


def rebuild(evaluation_dir: Path) -> Path:
    rows = []
    for results in sorted(evaluation_dir.rglob(RESULTS_NAME)):
        row = run_row(results.parent, evaluation_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No {RESULTS_NAME} under {evaluation_dir}")
    table = evaluation_dir / TABLE_NAME
    _write(table, rows)
    return table


def configured_evaluation_dir() -> Path:
    return repo_path(load_config()["paths"]["evaluation"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, default=None, help="Default: paths.evaluation from the config.")
    args = parser.parse_args()
    evaluation_dir = repo_path(args.evaluation) if args.evaluation else configured_evaluation_dir()
    table = rebuild(evaluation_dir)
    print(table)


if __name__ == "__main__":
    main()
