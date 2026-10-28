"""Save the config and a short meta file next to checkpoints and test runs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_NAME = "config.yaml"
META_NAME = "meta.yaml"

_EPOCH = re.compile(r"epoch(\d+)")
_BATCH = re.compile(r"batch(\d+)pct")
_EPOCH_SUFFIX = re.compile(r"-epoch\d+(?:-batch\d+pct)?$")


def copy_config(destination: Path, source: Path | None = None, *, replace: bool = False) -> None:
    """Copy a config into ``destination``.

    Training passes ``replace=True`` so the folder keeps the config of the
    process that last started there. A test run copies the config that sits
    beside the checkpoint when that file exists.
    """
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / CONFIG_NAME
    if target.is_file() and not replace:
        return
    origin = source if source is not None else CONFIG_PATH
    if not origin.is_file():
        raise SystemExit(f"Config not found: {origin}")
    shutil.copyfile(origin, target)


def write_meta(destination: Path, fields: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in fields.items() if value is not None}
    text = yaml.safe_dump(clean, sort_keys=False, allow_unicode=True)
    (destination / META_NAME).write_text(text, encoding="utf-8")


def checkpoint_fields(path: Path) -> dict:
    """Model name, filename, epoch, and batch percent parsed from a checkpoint."""
    resolved = path.resolve() if path.exists() else path
    name = resolved.name
    stem = resolved.stem
    model = _EPOCH_SUFFIX.sub("", stem) or stem
    fields = {"model": model, "checkpoint": name}
    epoch = _EPOCH.search(name)
    if epoch is not None:
        fields["epoch"] = int(epoch.group(1))
    batch = _BATCH.search(name)
    if batch is not None:
        fields["batch_percent"] = int(batch.group(1))
    return fields


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def config_for_checkpoint(folder: Path) -> Path:
    local = folder / CONFIG_NAME
    if local.is_file():
        return local
    return CONFIG_PATH


def test_record(
    kind: str,
    source: str,
    checkpoints_root: Path,
    base_model: str,
    language: str,
) -> tuple[dict, Path | None]:
    """Return meta fields and a config to copy.

    A Hugging Face id, or a directory that is not under the checkpoints root,
    gets a meta file with the model name and no config copy. A checkpoint gets
    the epoch and the config from its folder.
    """
    if kind == "hf":
        return {"model": source}, None
    path = Path(source)
    if kind == "dir":
        if not _inside(path, checkpoints_root):
            return {"model": path.name}, None
        fields = {"model": path.name, "checkpoint_dir": str(path.resolve()), "base_model": base_model, "language": language}
        newest = _newest_checkpoint(path)
        if newest is not None:
            fields.update(checkpoint_fields(newest))
        return fields, config_for_checkpoint(path)
    folder = path.parent
    fields = checkpoint_fields(path)
    fields["checkpoint_dir"] = str(folder.resolve())
    fields["base_model"] = base_model
    fields["language"] = language
    return fields, config_for_checkpoint(folder)


def _newest_checkpoint(folder: Path) -> Path | None:
    last = folder / "last.ckpt"
    if last.is_file():
        return last
    candidates = [path for path in folder.glob("*.ckpt") if path.is_file() and not path.is_symlink()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
