"""Load fine-tune and test settings from the repo ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class Settings:
    common_voice: Path
    checkpoints: Path
    model_id: str
    model_name: str
    language: str
    gpus: tuple[int, ...]
    train_batch_size: int
    testing_after_train_batch_size: int
    testing_batch_size: int
    learning_rate: float
    warmup_steps: int
    max_epochs: int
    num_workers: int
    precision: str
    seed: int
    gradient_clip: float
    weight_decay: float
    resume: bool
    baseline_eval: bool
    limit_train_batches: int | float
    limit_val_batches: int | float
    limit_test_batches: int | float
    checkpoint_every: float
    max_audio_seconds: float
    max_label_tokens: int


def _mapping(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"config.yaml: '{name}' must be a mapping")
    return value


def _require(section: dict, key: str, label: str):
    if key not in section:
        raise SystemExit(f"config.yaml missing {label}")
    return section[key]


def _bool(section: dict, key: str, label: str) -> bool:
    value = _require(section, key, label)
    if not isinstance(value, bool):
        raise SystemExit(f"config.yaml {label} must be true or false")
    return value


def _text(section: dict, key: str, label: str) -> str:
    value = _require(section, key, label)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"config.yaml {label} must be a non-empty string")
    return value.strip()


def _positive_int(section: dict, key: str, label: str) -> int:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"config.yaml {label} must be a positive integer")
    return value


def _non_negative_int(section: dict, key: str, label: str) -> int:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"config.yaml {label} must be a non-negative integer")
    return value


def _non_negative_float(section: dict, key: str, label: str) -> float:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SystemExit(f"config.yaml {label} must be a non-negative number")
    return float(value)


def _positive_float(section: dict, key: str, label: str) -> float:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"config.yaml {label} must be a positive number")
    return float(value)


def _fraction(section: dict, key: str, label: str) -> float:
    value = _positive_float(section, key, label)
    if value > 1:
        raise SystemExit(f"config.yaml {label} must be at most 1")
    return value


def _limit(section: dict, key: str, label: str) -> int | float:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"config.yaml {label} must be a positive number")
    return value


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{config_path} must be a mapping")

    paths = _mapping(loaded, "paths")
    model = _mapping(loaded, "model")
    gpus = _mapping(loaded, "gpus")
    batches = _mapping(loaded, "batch_size")
    training = _mapping(loaded, "training")
    data = _mapping(loaded, "data")

    selected = []
    for index in (0, 1):
        if _bool(gpus, f"nvidia{index}", f"gpus.nvidia{index}"):
            selected.append(index)
    if not selected:
        raise SystemExit("Enable at least one GPU in config.yaml (gpus.nvidia0, gpus.nvidia1).")

    return Settings(
        common_voice=Path(_text(paths, "common_voice", "paths.common_voice")),
        checkpoints=Path(_text(paths, "checkpoints", "paths.checkpoints")),
        model_id=_text(model, "id", "model.id"),
        model_name=_text(model, "name", "model.name"),
        language=_text(model, "language", "model.language"),
        gpus=tuple(selected),
        train_batch_size=_positive_int(batches, "train", "batch_size.train"),
        testing_after_train_batch_size=_positive_int(
            batches, "testing_after_train", "batch_size.testing_after_train"
        ),
        testing_batch_size=_positive_int(batches, "testing", "batch_size.testing"),
        learning_rate=_positive_float(training, "learning_rate", "training.learning_rate"),
        warmup_steps=_non_negative_int(training, "warmup_steps", "training.warmup_steps"),
        max_epochs=_positive_int(training, "max_epochs", "training.max_epochs"),
        num_workers=_non_negative_int(training, "num_workers", "training.num_workers"),
        precision=_text(training, "precision", "training.precision"),
        seed=_non_negative_int(training, "seed", "training.seed"),
        gradient_clip=_positive_float(training, "gradient_clip", "training.gradient_clip"),
        weight_decay=_non_negative_float(training, "weight_decay", "training.weight_decay"),
        resume=_bool(training, "resume", "training.resume"),
        baseline_eval=_bool(training, "baseline_eval", "training.baseline_eval"),
        limit_train_batches=_limit(training, "limit_train_batches", "training.limit_train_batches"),
        limit_val_batches=_limit(training, "limit_val_batches", "training.limit_val_batches"),
        limit_test_batches=_limit(training, "limit_test_batches", "training.limit_test_batches"),
        checkpoint_every=_fraction(training, "checkpoint_every", "training.checkpoint_every"),
        max_audio_seconds=_positive_float(data, "max_audio_seconds", "data.max_audio_seconds"),
        max_label_tokens=_positive_int(data, "max_label_tokens", "data.max_label_tokens"),
    )


@dataclass(frozen=True)
class StudentSettings:
    checkpoints: Path
    name: str
    base: str
    init: str
    marks: Path
    labels: Path
    filter_marks: bool
    accepted_mark: tuple[int, ...]
    voa: Path
    sample_rate: int
    language: str
    eval_data: str
    gpus: tuple[int, ...]
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_steps: int
    max_epochs: int
    num_workers: int
    precision: str
    seed: int
    gradient_clip: float
    weight_decay: float
    resume: bool
    baseline_eval: bool
    limit_train_batches: int | float
    limit_val_batches: int | float
    limit_test_batches: int | float
    checkpoint_every: float
    val_every: float
    max_audio_seconds: float
    max_label_tokens: int


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


def latest_labelling_file(output_dir: Path, teacher: Path, filename: str) -> Path:
    """Newest ``filename`` written for this teacher under ``output_dir``."""
    found = find_labelling_file(output_dir, teacher, filename)
    if found is None:
        raise SystemExit(
            f"No {filename} for {teacher.name} under {output_dir}. Run pseudo-labelling first."
        )
    return found


def _stored_path(section: dict, key: str, label: str, loaded: dict, filename: str) -> Path:
    raw = _text(section, key, label)
    if raw == "latest":
        section_labels = _mapping(loaded, "pseudo_labelling")
        return latest_labelling_file(
            _path(section_labels, "output_dir", "pseudo_labelling.output_dir"),
            _path(section_labels, "teacher_model", "pseudo_labelling.teacher_model"),
            filename,
        )
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _path(section: dict, key: str, label: str) -> Path:
    raw = Path(_text(section, key, label))
    if not raw.is_absolute():
        raw = ROOT / raw
    return raw


def _mark_list(section: dict, key: str, label: str) -> tuple[int, ...]:
    value = _require(section, key, label)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"config.yaml {label} must be a non-empty list of marks from 0 to 5")
    marks: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 5:
            raise SystemExit(f"config.yaml {label} must be a non-empty list of marks from 0 to 5")
        if item not in marks:
            marks.append(item)
    return tuple(marks)


def _gpus(section: dict, label: str) -> tuple[int, ...]:
    selected = []
    for index in (0, 1):
        key = f"nvidia{index}"
        if _bool(section, key, f"{label}.{key}"):
            selected.append(index)
    if not selected:
        raise SystemExit(f"Enable at least one GPU in config.yaml ({label}.nvidia0, {label}.nvidia1).")
    return tuple(selected)


def load_student_settings(path: Path | None = None) -> StudentSettings:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{config_path} must be a mapping")

    paths = _mapping(loaded, "paths")
    student = _mapping(loaded, "student")
    batches = _mapping(student, "batch_size")
    name = _text(student, "name", "student.name")
    if not name.startswith("student_"):
        raise SystemExit("config.yaml student.name must start with student_")
    accepted = _mark_list(student, "accepted_mark", "student.accepted_mark")
    eval_data = _text(student, "eval_data", "student.eval_data")
    if eval_data != "common_voice":
        raise SystemExit("config.yaml student.eval_data must be common_voice")
    filter_marks = _bool(student, "filter_marks", "student.filter_marks")
    labels = _stored_path(student, "labels", "student.labels", loaded, "labels.csv")
    if filter_marks:
        marks = _stored_path(student, "marks", "student.marks", loaded, "label_marks.csv")
    else:
        marks = Path()

    return StudentSettings(
        checkpoints=Path(_text(paths, "checkpoints", "paths.checkpoints")),
        name=name,
        base=_text(student, "base", "student.base"),
        init=_text(student, "init", "student.init"),
        marks=marks,
        labels=labels,
        filter_marks=filter_marks,
        accepted_mark=accepted,
        voa=_path(student, "voa", "student.voa"),
        sample_rate=_positive_int(student, "sample_rate", "student.sample_rate"),
        language=_text(student, "language", "student.language"),
        eval_data=eval_data,
        gpus=_gpus(_mapping(student, "gpus"), "student.gpus"),
        train_batch_size=_positive_int(batches, "train", "student.batch_size.train"),
        eval_batch_size=_positive_int(batches, "eval", "student.batch_size.eval"),
        learning_rate=_positive_float(student, "learning_rate", "student.learning_rate"),
        warmup_steps=_non_negative_int(student, "warmup_steps", "student.warmup_steps"),
        max_epochs=_positive_int(student, "max_epochs", "student.max_epochs"),
        num_workers=_non_negative_int(student, "num_workers", "student.num_workers"),
        precision=_text(student, "precision", "student.precision"),
        seed=_non_negative_int(student, "seed", "student.seed"),
        gradient_clip=_positive_float(student, "gradient_clip", "student.gradient_clip"),
        weight_decay=_non_negative_float(student, "weight_decay", "student.weight_decay"),
        resume=_bool(student, "resume", "student.resume"),
        baseline_eval=_bool(student, "baseline_eval", "student.baseline_eval"),
        limit_train_batches=_limit(student, "limit_train_batches", "student.limit_train_batches"),
        limit_val_batches=_limit(student, "limit_val_batches", "student.limit_val_batches"),
        limit_test_batches=_limit(student, "limit_test_batches", "student.limit_test_batches"),
        checkpoint_every=_fraction(student, "checkpoint_every", "student.checkpoint_every"),
        val_every=_fraction(student, "val_every", "student.val_every"),
        max_audio_seconds=_positive_float(student, "max_audio_seconds", "student.max_audio_seconds"),
        max_label_tokens=_positive_int(student, "max_label_tokens", "student.max_label_tokens"),
    )
