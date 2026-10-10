"""Checkpoint paths and mid-epoch saves for Whisper fine-tuning.

Lightning's ``trainer.save_checkpoint(..., weights_only=False)`` stores the
model weights together with ``optimizer_states`` and ``lr_schedulers``, so a
later ``Trainer.fit(ckpt_path=...)`` continues from the same step.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightning as L


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


class BatchFractionCheckpoint(L.Callback):
    """Save a full training checkpoint every ``fraction`` of an epoch, and at epoch end.

    Filenames carry the model name, the epoch, and the batch percentage, for
    example ``model-whisper-uk-epoch00-batch05pct.ckpt``.
    """

    def __init__(self, dirpath: Path, model_name: str, fraction: float = 0.05):
        self.dirpath = Path(dirpath)
        self.model_name = model_name
        self.fraction = fraction
        self._targets: dict[int, int] = {}

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

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        batches = self._batches_this_epoch(trainer)
        targets: dict[int, int] = {}
        for percent in range(5, 101, int(self.fraction * 100)):
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
        self._save(trainer, self.dirpath / filename)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        filename = f"{self.model_name}-epoch{trainer.current_epoch:02d}.ckpt"
        self._save(trainer, self.dirpath / filename)

    def _save(self, trainer: L.Trainer, path: Path) -> None:
        trainer.save_checkpoint(str(path), weights_only=False)
        if trainer.is_global_zero:
            last = self.dirpath / "last.ckpt"
            if last.is_symlink() or last.exists():
                last.unlink()
            last.symlink_to(path.name)
        trainer.strategy.barrier()
