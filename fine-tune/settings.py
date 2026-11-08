"""Load fine-tune and test settings from the project config.

The file is ``config.yaml`` in the repo root, or the one ``ASR_CONFIG`` names.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import CONFIG_PATH, configured_gpu, find_labelling_file, labelling_runs  # noqa: E402,F401
from dataset_roles import check_names  # noqa: E402


@dataclass(frozen=True)
class Settings:
    common_voice: Path
    fleurs: Path
    opentts: Path
    tg_voices: Path
    ukr_dialects: Path
    cv10_test: Path
    common_voice_splits: Path
    checkpoints: Path
    output_dir: Path
    evaluation_dir: Path
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
    checkpoint_every: float
    remove_mid_epoch_checkpoints: bool
    max_audio_seconds: float
    max_label_tokens: int
    train_data: tuple[str, ...]
    dev_data: tuple[str, ...]
    init: str | None
    test_data: tuple[str, ...]


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


def _optional_text(section: dict, key: str, label: str) -> str | None:
    value = section.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise SystemExit(f"config.yaml {label} must be a string")
    return value.strip()


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


def _checkpoint_every(loaded: dict) -> float:
    """Top-level ``checkpoint_every``: one value for every training stage."""
    value = _fraction(loaded, "checkpoint_every", "checkpoint_every")
    percent = value * 100
    if abs(percent - round(percent)) > 1e-9:
        raise SystemExit("config.yaml checkpoint_every must be a multiple of 0.01, e.g. 0.25")
    return value


def _fraction(section: dict, key: str, label: str) -> float:
    value = _positive_float(section, key, label)
    if value > 1:
        raise SystemExit(f"config.yaml {label} must be at most 1")
    return value


def _names(section: dict, key: str, label: str) -> tuple[str, ...]:
    value = _require(section, key, label)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise SystemExit(f"config.yaml {label} must be a non-empty list of dataset names")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"config.yaml {label} must contain dataset names")
        names.append(item.strip())
    if len(names) != len(set(names)):
        raise SystemExit(f"config.yaml {label} lists a dataset more than once")
    return tuple(names)


def _role_names(section: dict, role: str, label: str) -> tuple[str, ...]:
    names = _names(section, role, label)
    check_names(names, role, label)
    return names


def _extra_train(section: dict, label: str) -> tuple[str, ...]:
    """Optional labelled train sets mixed into the student's pseudo-labels."""
    if not section.get("extra_train"):
        return ()
    names = _names(section, "extra_train", label)
    check_names(names, "train", label)
    return names


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
    batches = _mapping(loaded, "batch_size")
    training = _mapping(loaded, "training")
    data = _mapping(loaded, "data")

    return Settings(
        common_voice=_path(paths, "common_voice", "paths.common_voice"),
        fleurs=_path(paths, "fleurs", "paths.fleurs"),
        opentts=_path(paths, "opentts", "paths.opentts"),
        tg_voices=_path(paths, "tg_voices", "paths.tg_voices"),
        ukr_dialects=_path(paths, "ukr_dialects", "paths.ukr_dialects"),
        cv10_test=_path(paths, "cv10_test", "paths.cv10_test"),
        common_voice_splits=_path(paths, "common_voice_splits", "paths.common_voice_splits"),
        checkpoints=_path(paths, "checkpoints", "paths.checkpoints"),
        output_dir=_path(paths, "fine_tune_output", "paths.fine_tune_output"),
        evaluation_dir=_path(paths, "evaluation", "paths.evaluation"),
        model_id=_text(model, "id", "model.id"),
        model_name=_text(model, "name", "model.name"),
        language=_text(model, "language", "model.language"),
        gpus=(configured_gpu(loaded),),
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
        checkpoint_every=_checkpoint_every(loaded),
        remove_mid_epoch_checkpoints=_bool(loaded, "remove_mid_epoch_checkpoints", "remove_mid_epoch_checkpoints"),
        max_audio_seconds=_positive_float(data, "max_audio_seconds", "data.max_audio_seconds"),
        max_label_tokens=_positive_int(data, "max_label_tokens", "data.max_label_tokens"),
        train_data=_role_names(data, "train", "data.train"),
        dev_data=_role_names(data, "dev", "data.dev"),
        test_data=_role_names(data, "test", "data.test"),
        init=_optional_text(training, "init", "training.init"),
    )


@dataclass(frozen=True)
class StudentSettings:
    checkpoints: Path
    output_dir: Path
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
    remove_mid_epoch_checkpoints: bool
    val_every: float
    max_audio_seconds: float
    max_label_tokens: int
    extra_train: tuple[str, ...]


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
        checkpoints=_path(paths, "checkpoints", "paths.checkpoints"),
        output_dir=_path(paths, "fine_tune_output", "paths.fine_tune_output"),
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
        # One GPU for the whole project: the top-level gpus section.
        gpus=(configured_gpu(loaded),),
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
        checkpoint_every=_checkpoint_every(loaded),
        remove_mid_epoch_checkpoints=_bool(loaded, "remove_mid_epoch_checkpoints", "remove_mid_epoch_checkpoints"),
        val_every=_fraction(student, "val_every", "student.val_every"),
        max_audio_seconds=_positive_float(student, "max_audio_seconds", "student.max_audio_seconds"),
        max_label_tokens=_positive_int(student, "max_label_tokens", "student.max_label_tokens"),
        extra_train=_extra_train(student, "student.extra_train"),
    )
