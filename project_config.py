"""Find and read the project config.

``ASR_CONFIG`` names the config file, for example ``config.vast.yaml``.
Without it the repo ``config.yaml`` is used. A relative path, in
``ASR_CONFIG`` or inside the config, resolves from the repo root.

``fine-tune/settings.py`` and ``pseudo-labelling/settings.py`` validate their
own sections. Scripts that need only a few values read them here.

    ASR_CONFIG=config.vast.yaml python main.py pipeline
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_ENV = "ASR_CONFIG"
DEFAULT_CONFIG = ROOT / "config.yaml"


def repo_path(raw: str | Path) -> Path:
    """An absolute path stays as is. A relative one resolves from the repo root."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def config_path() -> Path:
    raw = os.environ.get(CONFIG_ENV, "").strip()
    return repo_path(raw) if raw else DEFAULT_CONFIG


CONFIG_PATH = config_path()


def load_config(path: Path | None = None) -> dict:
    source = path or CONFIG_PATH
    if not source.is_file():
        raise SystemExit(f"Config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{source} must be a mapping")
    return loaded


def configured_gpu(loaded: dict | None = None) -> int:
    """The one GPU of every stage: the index of the single ``gpus.nvidiaN: true``."""
    config = loaded if loaded is not None else load_config()
    section = config.get("gpus")
    if not isinstance(section, dict):
        raise SystemExit(f"{CONFIG_PATH.name} missing gpus")
    selected = []
    for key, value in section.items():
        if not (key.startswith("nvidia") and key[6:].isdigit()) or not isinstance(value, bool):
            raise SystemExit(f"{CONFIG_PATH.name} gpus.{key} must be nvidiaN: true or false")
        if value:
            selected.append(int(key[6:]))
    if len(selected) != 1:
        raise SystemExit(f"Enable exactly one GPU in {CONFIG_PATH.name} (gpus.nvidiaN: true).")
    return selected[0]


def pin_gpu(loaded: dict | None = None) -> int:
    """Show only the configured GPU to this process. Call before CUDA starts.

    A process started by a distributed launcher (``LOCAL_RANK`` set) keeps the
    mask its parent set.
    """
    index = configured_gpu(loaded)
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return index


def config_value(loaded: dict, dotted: str):
    node = loaded
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"{CONFIG_PATH.name} missing {dotted}")
        node = node[part]
    return node


def config_text(loaded: dict, dotted: str) -> str:
    value = config_value(loaded, dotted)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{CONFIG_PATH.name} {dotted} must be a non-empty string")
    return value.strip()


def config_int(loaded: dict, dotted: str) -> int:
    value = config_value(loaded, dotted)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"{CONFIG_PATH.name} {dotted} must be a positive integer")
    return value


def config_file(loaded: dict, dotted: str) -> Path:
    return repo_path(config_text(loaded, dotted))


def labelling_runs(output_dir: Path, teacher: Path) -> list[Path]:
    """Numbered pseudo-label directories for this teacher, oldest first."""
    name = teacher.resolve().stem
    slug = "".join(character if character.isalnum() or character in "._-" else "-" for character in name)
    slug = slug.strip("-.") or "model"
    prefix = f"{slug}-"
    existing = []
    if output_dir.is_dir():
        for path in output_dir.glob(f"{prefix}*"):
            suffix = path.name.removeprefix(prefix)
            if path.is_dir() and suffix.isdigit():
                existing.append(path)
    existing.sort(key=lambda path: int(path.name.removeprefix(prefix)))
    return existing


def find_labelling_file(output_dir: Path, teacher: Path, filename: str) -> Path | None:
    """Newest ``filename`` for this teacher, or ``None`` when labelling has not written it."""
    for directory in reversed(labelling_runs(output_dir, teacher)):
        candidate = directory / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def configured_labelling_file(filename: str, loaded: dict | None = None) -> Path:
    """Newest ``filename`` for ``pseudo_labelling.teacher_model`` under ``pseudo_labelling.output_dir``."""
    config = loaded if loaded is not None else load_config()
    output_dir = config_file(config, "pseudo_labelling.output_dir")
    teacher = config_file(config, "pseudo_labelling.teacher_model")
    found = find_labelling_file(output_dir, teacher, filename)
    if found is None:
        raise SystemExit(f"No {filename} for {teacher.name} under {output_dir}. Run pseudo-labelling first.")
    return found
