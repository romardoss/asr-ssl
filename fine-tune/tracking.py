"""MLflow tracking for fine-tune, student, multi_step, and evaluation runs.

The ``mlflow`` section of the config turns it on. Without the section, or with
``enabled: false``, every function here does nothing. A tracking failure only
logs a warning: it never stops training or evaluation.

What lands in MLflow:

- a training run per checkpoint folder (``fine-tune.py``, ``fine-tune-student.py``).
  Lightning's own metrics go there by *step* (``train_loss_step``,
  ``train_loss_epoch``, ``lr-AdamW``, ``val_loss``, ``val_wer``, ``val_cer``, ...),
  so a dev score taken several times per epoch is a curve over steps.
  The same scores are also logged by *epoch* as ``epoch/train_loss`` and
  ``epoch/val_*`` (x axis = epoch number, from 0, like ``epoch00`` in checkpoint names).
- an evaluation run per ``evaluation/<model>-XX`` folder, in the experiment
  ``<experiment>-evaluation``: test metrics ``test/<metric>`` and
  ``test/<dataset>/<metric>``, tagged with ``train_run_id`` of the training run.
- a parent run per multi-step chain; its steps and evaluations nest under it.

A resumed training continues the same MLflow run: its id is stored as
``mlflow_run_id`` in the ``meta.yaml`` of the checkpoint folder.

    mlflow ui --backend-store-uri sqlite:///output/mlflow/mlflow.db
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import load_config, repo_path  # noqa: E402
from run_meta import read_meta  # noqa: E402

log = logging.getLogger("mlflow-tracking")

# multi_step.py puts the run id of the chain here, so every step and evaluation nests under it.
PARENT_ENV = "ASR_MLFLOW_PARENT"
EVALUATION_SUFFIX = "-evaluation"
CHAIN_RUN_FILE = "mlflow_run_id"
# Meta fields that change from checkpoint to checkpoint: tags or nothing, not params.
_NOT_PARAMS = {"note", "checkpoint", "epoch", "batch_percent", "resumed_from", "checkpoint_dir", "mlflow_run_id"}
_EVAL_TAGS = {"note", "git_commit", "evaluated_at"}
_BAD_NAME = re.compile(r"[^0-9A-Za-z_\-./ ]")


@dataclass(frozen=True)
class Settings:
    enabled: bool
    uri: str
    artifacts: Path
    experiment: str


def load_settings(config: dict | None = None) -> Settings:
    loaded = config if config is not None else load_config()
    section = loaded.get("mlflow")
    if not isinstance(section, dict):
        return Settings(False, "", Path(), "")
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SystemExit("config mlflow.enabled must be true or false")
    return Settings(
        enabled=enabled,
        uri=str(section.get("tracking_uri") or "sqlite:///output/mlflow/mlflow.db"),
        artifacts=repo_path(section.get("artifacts") or "output/mlflow/artifacts"),
        experiment=str(section.get("experiment") or "asr-whisper-uk"),
    )


def _uri(raw: str) -> str:
    """``sqlite:///relative.db`` resolves from the repo root; other URIs stay as they are."""
    prefix = "sqlite:///"
    if raw.startswith(prefix) and not raw.startswith(prefix + "/"):
        path = repo_path(raw[len(prefix):])
        path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + str(path)
    return raw


def _open(settings: Settings, experiment: str):
    """``(client, tracking uri, experiment id)``; the experiment is created on first use."""
    from mlflow.tracking import MlflowClient

    uri = _uri(settings.uri)
    client = MlflowClient(tracking_uri=uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        settings.artifacts.mkdir(parents=True, exist_ok=True)
        try:
            return client, uri, client.create_experiment(experiment, artifact_location=settings.artifacts.resolve().as_uri())
        except Exception:
            found = client.get_experiment_by_name(experiment)  # another process created it first
            if found is None:
                raise
    return client, uri, found.experiment_id


def _name(key: str) -> str:
    return _BAD_NAME.sub("_", str(key))


def _text(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _params(values: dict) -> dict[str, str]:
    return {_name(key): _text(value)[:500] for key, value in values.items() if value is not None}


def _tags(values: dict) -> dict[str, str]:
    return {_name(key) if not key.startswith("mlflow.") else key: _text(value)[:5000] for key, value in values.items() if value is not None}


def git_commit() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return done.stdout.strip() or None


def _run_tags(name: str, kind: str, extra: dict, *, nest: bool = True) -> dict[str, str]:
    tags = {"mlflow.runName": name, "type": kind, "git_commit": git_commit()}
    tags.update({key: value for key, value in extra.items() if value is not None})
    parent = os.environ.get(PARENT_ENV, "").strip()
    if nest and parent:
        tags["mlflow.parentRunId"] = parent
    return _tags(tags)


def _log_params(client, run_id: str, params: dict[str, str]) -> None:
    """Params never change once set, so a resumed run only adds the new ones."""
    from mlflow.entities import Param

    known = client.get_run(run_id).data.params
    fresh = [Param(key, value) for key, value in params.items() if key not in known]
    for start in range(0, len(fresh), 100):
        client.log_batch(run_id, params=fresh[start : start + 100])


def _log_metrics(client, run_id: str, values: dict, step: int, timestamp: int | None = None) -> None:
    from mlflow.entities import Metric

    stamp = timestamp if timestamp is not None else int(time.time() * 1000)
    metrics = []
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics.append(Metric(_name(key), number, stamp, int(step)))
    for start in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[start : start + 1000])


def _alive(client, run_id: str, experiment_id: str) -> bool:
    try:
        info = client.get_run(run_id).info
    except Exception:
        return False
    return info.experiment_id == experiment_id and info.lifecycle_stage == "active"


def config_params(config: dict, stage: str) -> dict:
    """Training settings from a config that are worth comparing between runs."""
    student = stage == "student"
    section = config.get("student" if student else "training") or {}
    batches = (section if student else config).get("batch_size") or {}
    data = config.get("data") or {}
    return {
        "batch_size": batches.get("train"),
        "warmup_steps": section.get("warmup_steps"),
        "weight_decay": section.get("weight_decay"),
        "gradient_clip": section.get("gradient_clip"),
        "checkpoint_every": config.get("checkpoint_every"),
        "val_every": section.get("val_every") if student else None,
        "dev_data": None if student else data.get("dev"),
        "test_data": None if student else data.get("test"),
    }


def clip_params(splits: dict) -> dict:
    """Clip counts of each split (and of each dataset in it), after the leak filter."""
    counts = {}
    for name, split in splits.items():
        counts[f"clips.{name}"] = len(split)
        for part in split.parts:
            counts[f"clips.{name}.{part.source}"] = len(part)
    return counts


class Tracked:
    """One open MLflow run. Every method logs a warning instead of raising."""

    def __init__(self, client, uri: str, run_id: str):
        self.client = client
        self.uri = uri
        self.run_id = run_id

    def logger(self):
        """The Lightning logger that writes the training metrics of this run by step."""
        from lightning.pytorch.loggers import MLFlowLogger

        return MLFlowLogger(run_id=self.run_id, tracking_uri=self.uri, log_model=False)

    def callback(self):
        """The Lightning callback that adds the per-epoch series."""
        return _epoch_callback(self.client, self.run_id)

    def log_files(self, paths, artifact_path: str | None = None) -> None:
        for path in paths:
            try:
                if Path(path).is_file():
                    self.client.log_artifact(self.run_id, str(path), artifact_path)
            except Exception as error:
                log.warning("MLflow artifact %s not saved: %s", path, error)

    def log_dir(self, path, artifact_path: str | None = None) -> None:
        try:
            if Path(path).is_dir():
                self.client.log_artifacts(self.run_id, str(path), artifact_path)
        except Exception as error:
            log.warning("MLflow artifacts %s not saved: %s", path, error)


def _epoch_callback(client, run_id: str):
    import lightning as L

    class EpochMetrics(L.Callback):
        """``epoch/train_loss`` and the latest ``epoch/val_*`` at the end of each epoch."""

        def __init__(self) -> None:
            self.total = None
            self.count = 0

        def on_train_epoch_start(self, trainer, module) -> None:
            self.total = None
            self.count = 0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is None:
                return
            loss = loss.detach().float()
            self.total = loss if self.total is None else self.total + loss
            self.count += 1

        def on_train_epoch_end(self, trainer, module) -> None:
            if not trainer.is_global_zero:
                return
            values = {}
            if self.count:
                values["epoch/train_loss"] = float(self.total) / self.count
            for key, value in trainer.callback_metrics.items():
                if key.startswith("val_"):
                    values[f"epoch/{key}"] = float(value)
            try:
                _log_metrics(client, run_id, values, trainer.current_epoch)
            except Exception as error:
                log.warning("MLflow epoch metrics not saved: %s", error)

    return EpochMetrics()


def start_training(name: str, folder: Path, params: dict, tags: dict, *, resumed: bool) -> Tracked | None:
    """Open the run of a training: new, or the one stored in ``folder`` when resuming."""
    settings = load_settings()
    if not settings.enabled or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    try:
        client, uri, experiment_id = _open(settings, settings.experiment)
        stored = read_meta(folder).get("mlflow_run_id") if resumed else None
        if stored and _alive(client, str(stored), experiment_id):
            run_id = str(stored)
            client.update_run(run_id, status="RUNNING")
            for key, value in _run_tags(name, "training", tags).items():
                client.set_tag(run_id, key, value)
        else:
            run_id = client.create_run(experiment_id, tags=_run_tags(name, "training", tags)).info.run_id
        _log_params(client, run_id, _params(params))
        log.info("MLflow run %s in experiment %s (%s)", run_id, settings.experiment, uri)
        return Tracked(client, uri, run_id)
    except Exception as error:
        log.warning("MLflow is off for this run: %s", error)
        return None


def _stamp(text) -> int | None:
    try:
        return int(datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    except ValueError:
        return None


def _existing(client, experiment_id: str, source_dir: str) -> str | None:
    found = client.search_runs([experiment_id], filter_string=f"tags.source_dir = '{source_dir}'", max_results=1)
    return found[0].info.run_id if found else None


def log_evaluation(run_dir: Path) -> str | None:
    """Log ``evaluation/<model>-XX`` (its results.json and meta.yaml) as one run. Returns the run id."""
    settings = load_settings()
    if not settings.enabled:
        return None
    try:
        results = Path(run_dir) / "results.json"
        if not results.is_file():
            return None
        payload = json.loads(results.read_text(encoding="utf-8"))
        meta = read_meta(run_dir)
        client, _uri_, experiment_id = _open(settings, settings.experiment + EVALUATION_SUFFIX)
        existing = _existing(client, experiment_id, Path(run_dir).name)
        if existing:
            log.info("MLflow already has %s: run %s", Path(run_dir).name, existing)
            return existing

        started = _stamp(meta.get("evaluated_at"))
        train_id = None
        trained_in = meta.get("checkpoint_dir")
        if trained_in and Path(trained_in).is_dir():
            train_id = read_meta(Path(trained_in)).get("mlflow_run_id")
        tags = {
            "source_dir": Path(run_dir).name,
            "train_run_id": train_id,
            **{key: meta.get(key) for key in _EVAL_TAGS},
        }
        run = client.create_run(experiment_id, start_time=started, tags=_run_tags(Path(run_dir).name, "evaluation", tags))
        run_id = run.info.run_id

        params = {key: value for key, value in meta.items() if key not in _NOT_PARAMS | _EVAL_TAGS}
        params.update(
            {
                "split": payload.get("split"),
                "kind": payload.get("kind"),
                "eval_batch_size": payload.get("eval_batch_size"),
                "model_source": payload.get("model"),
                "checkpoint": meta.get("checkpoint"),
                "epoch": meta.get("epoch"),
                "batch_percent": meta.get("batch_percent"),
            }
        )
        for name, clips in (payload.get("clips") or {}).items():
            params[f"clips.{name}"] = clips
        _log_params(client, run_id, _params(params))

        values = {f"test/{key}": value for key, value in (payload.get("metrics") or {}).items()}
        values["test/loss"] = payload.get("loss")
        for name, scored in (payload.get("test_sets") or {}).items():
            if name != "combined_weighted":
                values.update({f"test/{name}/{key}": value for key, value in scored.items()})
        _log_metrics(client, run_id, values, 0, started)

        tracked = Tracked(client, _uri(settings.uri), run_id)
        tracked.log_files([results, Path(run_dir) / "meta.yaml", Path(run_dir) / "config.yaml", Path(run_dir) / "test.log"])
        tracked.log_dir(Path(run_dir) / "plots", "plots")
        client.set_terminated(run_id, "FINISHED", end_time=started)
        log.info("MLflow evaluation run %s", run_id)
        return run_id
    except Exception as error:
        log.warning("MLflow evaluation not saved: %s", error)
        return None


def log_training_version(version_dir: Path, model: str) -> str | None:
    """Log an old ``<fine_tune_output>/<model>/version_N`` (metrics.csv, meta.yaml) as one run."""
    import pandas as pd
    import yaml

    settings = load_settings()
    if not settings.enabled:
        return None
    folder = Path(version_dir)
    metrics_file = folder / "metrics.csv"
    if not metrics_file.is_file():
        return None
    try:
        client, uri, experiment_id = _open(settings, settings.experiment)
        source_dir = f"{model}/{folder.name}"
        existing = _existing(client, experiment_id, source_dir)
        if existing:
            log.info("MLflow already has %s: run %s", source_dir, existing)
            return existing

        meta = read_meta(folder)
        name = Path(meta["checkpoint_dir"]).name if meta.get("checkpoint_dir") else source_dir
        stamp = int(metrics_file.stat().st_mtime * 1000)
        tags = {
            "source_dir": source_dir,
            "model": meta.get("model", model),
            "stage": meta.get("stage"),
            "chain": meta.get("chain"),
            "step": meta.get("step"),
            "note": meta.get("note"),
            "checkpoint_dir": meta.get("checkpoint_dir"),
        }
        run_id = client.create_run(
            experiment_id, start_time=stamp, tags=_run_tags(name, "training", tags, nest=False)
        ).info.run_id

        params = {key: value for key, value in meta.items() if key not in _NOT_PARAMS and key not in tags}
        config_file = folder / "config.yaml"
        if config_file.is_file():
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                params.update(config_params(loaded, str(meta.get("stage", ""))))
        _log_params(client, run_id, _params(params))

        frame = pd.read_csv(metrics_file)
        for _, row in frame.iterrows():
            if pd.isna(row.get("step")):
                continue
            values = {key: value for key, value in row.items() if key not in {"epoch", "step"} and not pd.isna(value)}
            _log_metrics(client, run_id, values, int(row["step"]), stamp)
        by_epoch = frame.dropna(subset=["epoch"]).copy()
        by_epoch["epoch"] = by_epoch["epoch"].astype(int)
        for epoch, group in by_epoch.groupby("epoch"):
            values = {}
            for column in group.columns:
                if column == "train_loss_epoch":
                    key = "epoch/train_loss"
                elif column.startswith("val_"):
                    key = f"epoch/{column}"
                else:
                    continue
                present = group[column].dropna()
                if not present.empty:
                    values[key] = present.iloc[-1]
            _log_metrics(client, run_id, values, int(epoch), stamp)

        tracked = Tracked(client, uri, run_id)
        tracked.log_files([metrics_file, folder / "meta.yaml", config_file])
        tracked.log_dir(folder / "plots", "plots")
        client.set_terminated(run_id, "FINISHED", end_time=stamp)
        log.info("MLflow training run %s", run_id)
        return run_id
    except Exception as error:
        log.warning("MLflow training run %s not saved: %s", version_dir, error)
        return None


def chain_run(name: str, folder: Path, params: dict) -> str | None:
    """The parent run of a multi-step chain; a rerun of the chain reuses it."""
    settings = load_settings()
    if not settings.enabled:
        return None
    try:
        client, _uri_, experiment_id = _open(settings, settings.experiment)
        marker = Path(folder) / CHAIN_RUN_FILE
        stored = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if stored and _alive(client, stored, experiment_id):
            run_id = stored
            client.update_run(run_id, status="RUNNING")
        else:
            run_id = client.create_run(
                experiment_id, tags=_run_tags(name, "chain", {"chain": name}, nest=False)
            ).info.run_id
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(run_id + "\n", encoding="utf-8")
        _log_params(client, run_id, _params(params))
        return run_id
    except Exception as error:
        log.warning("MLflow chain run not saved: %s", error)
        return None


def finish_run(run_id: str | None, ok: bool) -> None:
    if not run_id:
        return
    settings = load_settings()
    try:
        client, _uri_, _experiment = _open(settings, settings.experiment)
        client.set_terminated(run_id, "FINISHED" if ok else "FAILED")
    except Exception as error:
        log.warning("MLflow run %s not closed: %s", run_id, error)
