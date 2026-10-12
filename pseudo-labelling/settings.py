"""Labelling settings from the repo ``config.yaml``. Nothing is defaulted in code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class LabellingSettings:
    teacher_model: Path
    base_model: str
    voa: Path
    ukrainian_meta: Path
    timestamps: Path
    sample_rate: int
    batch_size: int
    language: str
    task: str
    precision: str
    resume: bool
    output_dir: Path
    gpus: tuple[int, ...]
    max_audio_seconds: float


def _mapping(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"config.yaml: '{name}' must be a mapping")
    return value


def _require(section: dict, key: str, label: str):
    if key not in section:
        raise SystemExit(f"config.yaml missing {label}")
    return section[key]


def _text(section: dict, key: str, label: str) -> str:
    value = _require(section, key, label)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"config.yaml {label} must be a non-empty string")
    return value.strip()


def _bool(section: dict, key: str, label: str) -> bool:
    value = _require(section, key, label)
    if not isinstance(value, bool):
        raise SystemExit(f"config.yaml {label} must be true or false")
    return value


def _positive_int(section: dict, key: str, label: str) -> int:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"config.yaml {label} must be a positive integer")
    return value


def _positive_float(section: dict, key: str, label: str) -> float:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"config.yaml {label} must be a positive number")
    return float(value)


def _path(section: dict, key: str, label: str) -> Path:
    raw = Path(_text(section, key, label))
    if not raw.is_absolute():
        raw = ROOT / raw
    return raw


def load_labelling(path: Path | None = None) -> LabellingSettings:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{config_path} must be a mapping")

    section = _mapping(loaded, "pseudo_labelling")
    model = _mapping(loaded, "model")
    gpus = _mapping(loaded, "gpus")
    data = _mapping(loaded, "data")
    selected = []
    for index in (0, 1):
        key = f"nvidia{index}"
        value = _require(gpus, key, f"gpus.{key}")
        if not isinstance(value, bool):
            raise SystemExit(f"config.yaml gpus.{key} must be true or false")
        if value:
            selected.append(index)
    if not selected:
        raise SystemExit("Enable at least one GPU in config.yaml (gpus.nvidia0, gpus.nvidia1).")

    return LabellingSettings(
        teacher_model=_path(section, "teacher_model", "pseudo_labelling.teacher_model"),
        base_model=_text(model, "id", "model.id"),
        voa=_path(section, "voa", "pseudo_labelling.voa"),
        ukrainian_meta=_path(section, "ukrainian_meta", "pseudo_labelling.ukrainian_meta"),
        timestamps=_path(section, "timestamps", "pseudo_labelling.timestamps"),
        sample_rate=_positive_int(section, "sample_rate", "pseudo_labelling.sample_rate"),
        batch_size=_positive_int(section, "batch_size", "pseudo_labelling.batch_size"),
        language=_text(section, "language", "pseudo_labelling.language"),
        task=_text(section, "task", "pseudo_labelling.task"),
        precision=_text(section, "precision", "pseudo_labelling.precision"),
        resume=_bool(section, "resume", "pseudo_labelling.resume"),
        output_dir=_path(section, "output_dir", "pseudo_labelling.output_dir"),
        gpus=tuple(selected),
        max_audio_seconds=_positive_float(data, "max_audio_seconds", "data.max_audio_seconds"),
    )
