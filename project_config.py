"""Find and read the project config.

``ASR_CONFIG`` names the config file, for example ``config.vast.yaml``.
Without it the repo ``config.yaml`` is used. A relative path, in
``ASR_CONFIG`` or inside the config, resolves from the repo root.

``fine-tune/settings.py`` and ``pseudo-labelling/settings.py`` validate their
own sections. Scripts that need only a few values read them here.

``load_config`` returns the config with three conveniences already applied, so
every reader sees one complete mapping:

1. ``extends: config.yaml`` at the top of a file: that file is the base, and this
   one lists only what differs. Mappings merge key by key, lists and values are
   replaced (``config.vast.yaml`` works this way).
2. ``${paths.data_root}`` in a string: replaced by the value of that key
   (any dotted key works). The override of ``paths.data_root`` in an extending
   file therefore moves every path built from it.
3. ``student`` takes ``SHARED_TRAINING_KEYS`` from ``training`` (and
   ``student.batch_size.train`` from ``batch_size.train``) unless it sets them itself.

    ASR_CONFIG=config.vast.yaml python main.py pipeline
"""

from __future__ import annotations

import os
import re
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


# Settings the student run shares with the teacher run (``training``), unless ``student`` sets its own.
SHARED_TRAINING_KEYS = (
    "learning_rate",
    "warmup_steps",
    "max_epochs",
    "num_workers",
    "precision",
    "seed",
    "gradient_clip",
    "weight_decay",
    "accumulate_grad_batches",
    "resume",
    "val_every",
    "limit_train_batches",
    "limit_val_batches",
)
EXTENDS_KEY = "extends"
_VARIABLE = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


def _read_mapping(source: Path) -> dict:
    if not source.is_file():
        raise SystemExit(f"Config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{source} must be a mapping")
    return loaded


def _merge(base: dict, override: dict) -> dict:
    """Mappings merge key by key; lists and plain values of ``override`` replace those of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _with_base(source: Path, seen: tuple[Path, ...] = ()) -> dict:
    loaded = _read_mapping(source)
    parent = loaded.pop(EXTENDS_KEY, None)
    if parent is None:
        return loaded
    if not isinstance(parent, str) or not parent.strip():
        raise SystemExit(f"{source.name} {EXTENDS_KEY} must be a config file path")
    base = repo_path(parent.strip())
    if base.resolve() in {path.resolve() for path in (*seen, source)}:
        raise SystemExit(f"{source.name} {EXTENDS_KEY} loops back to {base.name}")
    return _merge(_with_base(base, (*seen, source)), loaded)


def _interpolate(config: dict, name: str) -> dict:
    def lookup(dotted: str):
        node = config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise SystemExit(f"{name}: ${{{dotted}}} points to a key that does not exist")
            node = node[part]
        if isinstance(node, (dict, list)) or node is None:
            raise SystemExit(f"{name}: ${{{dotted}}} must point to a string or a number")
        return str(node)

    def expand(value):
        if isinstance(value, str):
            for _ in range(8):  # a variable may itself hold a variable
                replaced = _VARIABLE.sub(lambda match: lookup(match.group(1)), value)
                if replaced == value:
                    return value
                value = replaced
            raise SystemExit(f"{name}: variables in {value!r} do not resolve (a loop?)")
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(config)


def _inherit_student(config: dict) -> None:
    training = config.get("training")
    student = config.get("student")
    if not isinstance(training, dict) or not isinstance(student, dict):
        return
    for key in SHARED_TRAINING_KEYS:
        if key in training and key not in student:
            student[key] = training[key]
    teacher_batch = config.get("batch_size")
    if isinstance(teacher_batch, dict) and "train" in teacher_batch:
        batch = student.setdefault("batch_size", {})
        if isinstance(batch, dict) and "train" not in batch:
            batch["train"] = teacher_batch["train"]


def load_config(path: Path | None = None) -> dict:
    """The config as one complete mapping: base file merged, variables expanded, student defaults filled."""
    source = path or CONFIG_PATH
    config = _interpolate(_with_base(source), source.name)
    _inherit_student(config)
    return config


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
