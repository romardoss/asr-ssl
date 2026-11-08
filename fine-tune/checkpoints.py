"""Checkpoint paths and mid-epoch saves for Whisper fine-tuning.

Lightning's ``trainer.save_checkpoint(..., weights_only=False)`` stores the
model weights together with ``optimizer_states`` and ``lr_schedulers``, so a
later ``Trainer.fit(ckpt_path=...)`` continues from the same step. When a run
finishes, ``remove_mid_epoch`` deletes its mid-epoch saves and keeps the end
of every epoch.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import lightning as L

from run_meta import copy_config, write_meta


MID_EPOCH = re.compile(r"-epoch\d+-batch\d+pct\.ckpt$")


def remove_mid_epoch(directory: Path, dry_run: bool = False) -> str | None:
    """Delete the mid-epoch checkpoints of a finished run and say what was done.

    The end of every epoch and ``last.ckpt`` stay. Nothing is deleted while
    ``last.ckpt`` points to a mid-epoch file: that run is unfinished and may
    still resume from it. Returns None when there was nothing to delete.
    """
    directory = Path(directory)
    last = directory / "last.ckpt"
    if last.is_symlink() and MID_EPOCH.search(last.readlink().name):
        return f"kept mid-epoch checkpoints in {directory}: last.ckpt points to one"
    files = sorted(path for path in directory.glob("*.ckpt") if not path.is_symlink() and MID_EPOCH.search(path.name))
    if not files:
        return None
    size = sum(path.stat().st_size for path in files) / 1e9
    if dry_run:
        return f"would delete {len(files)} mid-epoch checkpoints ({size:.1f} GB) in {directory}"
    for path in files:
        path.unlink()
    return f"deleted {len(files)} mid-epoch checkpoints ({size:.1f} GB) in {directory}"


def latest_checkpoint(directory: Path) -> Path | None:
    """Return the checkpoint training should resume from, if one exists.

    ``last.ckpt`` is the symlink updated on every save. If it is absent, the
    newest ``*.ckpt`` in the directory is used instead.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    last = directory / "last.ckpt"
    if last.is_file():
        return last
    candidates = [path for path in directory.glob("*.ckpt") if path.is_file() and not path.is_symlink()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resume_checkpoint(directory: Path, enabled: bool) -> Path | None:
    if not enabled:
        return None
    return latest_checkpoint(directory)


def checkpoint_filename(path: Path) -> str:
    """Filename of a checkpoint, following a ``last.ckpt`` symlink."""
    if path.is_symlink():
        return path.readlink().name
    return path.name


def is_final_epoch_checkpoint(path: Path, max_epochs: int) -> bool:
    """True when ``path`` is the save from the last epoch, not a mid-epoch save."""
    match = re.search(r"-epoch(\d+)\.ckpt$", checkpoint_filename(path))
    return match is not None and int(match.group(1)) == max_epochs - 1


def _numbered_dirs(root: Path, model_name: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")
    found: list[tuple[int, Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            match = pattern.match(path.name)
            if path.is_dir() and match:
                found.append((int(match.group(1)), path))
    found.sort()
    return found


def student_checkpoint_dirs(root: Path, model_name: str) -> list[Path]:
    """Unnumbered ``<model_name>`` first, then ``_01``, ``_02``, and so on."""
    root = Path(root)
    numbered = [path for _number, path in _numbered_dirs(root, model_name)]
    plain = root / model_name
    if plain.is_dir():
        return [plain, *numbered]
    return numbered


def next_student_checkpoint_dir(root: Path, model_name: str) -> Path:
    root = Path(root)
    found = _numbered_dirs(root, model_name)
    number = found[-1][0] + 1 if found else 1
    return root / f"{model_name}_{number:02d}"


def student_checkpoint_run(
    root: Path, model_name: str, resume: bool, max_epochs: int
) -> tuple[Path, Path | None]:
    """Return the student directory and the checkpoint to resume, if any.

    ``resume`` continues only an unfinished newest directory. A directory whose
    ``last.ckpt`` is ``epoch{max_epochs - 1}.ckpt`` is a finished experiment, so
    the next run starts in a new numbered directory from the initial weights.
    ``resume`` false does that too. The unnumbered directory is the oldest run.
    """
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be positive, got {max_epochs}")
    directories = student_checkpoint_dirs(root, model_name)
    if resume and directories:
        newest = directories[-1]
        checkpoint = latest_checkpoint(newest)
        if checkpoint is None or not is_final_epoch_checkpoint(checkpoint, max_epochs):
            return newest, checkpoint
    return next_student_checkpoint_dir(root, model_name), None


def latest_student_checkpoint(root: Path, model_name: str) -> Path | None:
    """Newest student checkpoint, including one from a finished run."""
    for directory in reversed(student_checkpoint_dirs(root, model_name)):
        checkpoint = latest_checkpoint(directory)
        if checkpoint is not None:
            return checkpoint
    return None


def numbered_checkpoint_dir(root: Path, model_name: str, resume: bool) -> Path:
    """Return ``<root>/<model_name>_01``, then ``_02``, and so on.

    The unnumbered ``<model_name>`` directory is left untouched. With
    ``resume``, training continues in the newest numbered directory that
    already holds a checkpoint.
    """
    root = Path(root)
    found = _numbered_dirs(root, model_name)
    if resume:
        for _number, path in reversed(found):
            if latest_checkpoint(path) is not None:
                return path
        if found and latest_checkpoint(found[-1][1]) is None:
            return found[-1][1]
    number = found[-1][0] + 1 if found else 1
    return root / f"{model_name}_{number:02d}"


class BatchFractionCheckpoint(L.Callback):
    """Save a full training checkpoint every ``fraction`` of an epoch, and at epoch end.

    Filenames carry the model name, the epoch, and the batch percentage, for
    example ``model-whisper-uk-epoch00-batch25pct.ckpt``.
    """

    def __init__(self, dirpath: Path, model_name: str, fraction: float = 0.05, meta: dict | None = None):
        self.dirpath = Path(dirpath)
        self.model_name = model_name
        self.fraction = fraction
        self.meta = dict(meta or {})
        self.metrics_dir: Path | None = None
        self._targets: dict[int, int] = {}

    def _publish(self, fields: dict) -> None:
        payload = dict(fields)
        payload["checkpoint_dir"] = str(self.dirpath.resolve())
        copy_config(self.dirpath, replace=True)
        write_meta(self.dirpath, payload)
        if self.metrics_dir is not None:
            copy_config(self.metrics_dir, replace=True)
            write_meta(self.metrics_dir, payload)
        self.meta = payload

    def _batches_this_epoch(self, trainer: L.Trainer) -> int:
        count = trainer.num_training_batches
        if isinstance(count, list):
            count = sum(count)
        count = int(count)
        if count < 1 or math.isinf(count):
            raise RuntimeError(f"Cannot place batch checkpoints with num_training_batches={count}")
        return count

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            self._publish(self.meta)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        batches = self._batches_this_epoch(trainer)
        targets: dict[int, int] = {}
        # checkpoint_every 0.25 saves at 25, 50, and 75 %. The end of the epoch has its own save.
        step = round(self.fraction * 100)
        for percent in range(step, 100, step):
            batch_number = max(1, int(round(batches * percent / 100)))
            batch_number = min(batch_number, batches)
            targets.setdefault(batch_number, percent)
        self._targets = targets

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        percent = self._targets.get(batch_idx + 1)
        if percent is None:
            return
        filename = f"{self.model_name}-epoch{trainer.current_epoch:02d}-batch{percent:02d}pct.ckpt"
        self._save(trainer, self.dirpath / filename, trainer.current_epoch, percent)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        filename = f"{self.model_name}-epoch{trainer.current_epoch:02d}.ckpt"
        self._save(trainer, self.dirpath / filename, trainer.current_epoch, None)

    def _save(self, trainer: L.Trainer, path: Path, epoch: int, batch_percent: int | None) -> None:
        trainer.save_checkpoint(str(path), weights_only=False)
        if trainer.is_global_zero:
            last = self.dirpath / "last.ckpt"
            if last.is_symlink() or last.exists():
                last.unlink()
            last.symlink_to(path.name)
            payload = dict(self.meta)
            payload["checkpoint"] = path.name
            payload["epoch"] = int(epoch)
            if batch_percent is None:
                payload.pop("batch_percent", None)
            else:
                payload["batch_percent"] = int(batch_percent)
            self._publish(payload)
        trainer.strategy.barrier()
