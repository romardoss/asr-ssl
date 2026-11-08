"""Shared pieces of the leak-clean scripts: config, text normalization, output files."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import CONFIG_PATH, config_file, config_text, configured_labelling_file, load_config, repo_path

OUTPUT_DIR = ROOT / "output" / "leak-clean"
METADATA_NAME = "metadata.parquet"
COVERAGE_NAME = "coverage.csv"
DUPLICATES_NAME = "duplicates.csv"
UNITS_NAME = "units.csv"
AUDIO_MATCHES_NAME = "audio_matches.csv"
PLOTS_DIRNAME = "plots"

DATASET_ORDER = ("common_voice", "fleurs", "opentts", "tg_voices", "ukr_dialects", "cv10_test", "voa")
SPLIT_ORDER = ("train", "dev", "test")
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'"})
_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_LOOSE_APOSTROPHES = re.compile(r"(?<!\w)'|'(?!\w)")
_SPACES = re.compile(r"\s+")

__all__ = [
    "AUDIO_MATCHES_NAME",
    "AUDIO_SUFFIXES",
    "CONFIG_PATH",
    "COVERAGE_NAME",
    "DATASET_ORDER",
    "DUPLICATES_NAME",
    "METADATA_NAME",
    "OUTPUT_DIR",
    "PLOTS_DIRNAME",
    "ROOT",
    "SPLIT_ORDER",
    "UNITS_NAME",
    "audio_hash",
    "config_file",
    "config_text",
    "configured_labelling_file",
    "load_config",
    "normalize_text",
    "output_dir",
    "repo_path",
    "unit_key",
]


def normalize_text(text) -> str:
    """Lowercase, one apostrophe, no punctuation, single spaces.

    This is ``normalize_text`` from ``fine-tune/fine-tune.py`` (the WER
    metric) plus two steps for matching: every apostrophe variant becomes
    ``'``, and an apostrophe that is not inside a word is dropped, so a
    quoted 'слово' matches слово.
    """
    if text is None or text != text:  # None or NaN
        return ""
    text = str(text).lower().translate(_APOSTROPHES)
    text = _PUNCTUATION.sub(" ", text)
    text = _LOOSE_APOSTROPHES.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def audio_hash(data) -> str:
    """SHA-256 of the encoded audio bytes as stored, or "" when there are none."""
    if not data:
        return ""
    return hashlib.sha256(data).hexdigest()


def unit_key(unit: str) -> tuple[int, int, str]:
    """Sort ``dataset/split`` names in dataset order, then train, dev, test."""
    dataset, _, split = unit.partition("/")
    dataset_rank = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
    split_rank = SPLIT_ORDER.index(split) if split in SPLIT_ORDER else len(SPLIT_ORDER)
    return dataset_rank, split_rank, unit


def output_dir(raw: Path | None) -> Path:
    return OUTPUT_DIR if raw is None else repo_path(raw)
