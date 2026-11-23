"""VOA pseudo-label slices: which rows train the student, and their decoded-audio cache.

The cache (``student.voa_cache``) holds one ``<ID>.npy`` per slice: mono int16
at ``student.sample_rate``, already cut and capped at ``student.max_audio_seconds``.
``fine-tune/cache_voa.py`` writes it; the student reads a cached slice instead of
decoding its mp3 with ffmpeg. ``cache.json`` in the cache root records the
sample rate and the cap, so a cache made with other values is not used.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Appended: pseudo-labelling/settings.py must not shadow fine-tune/settings.py.
if str(ROOT / "pseudo-labelling") not in sys.path:
    sys.path.append(str(ROOT / "pseudo-labelling"))
from preprocess import parse_timestamp  # noqa: E402

# int16 <-> float32 [-1, 1)
INT16_SCALE = 32768.0
CACHE_META = "cache.json"


@dataclass(frozen=True)
class SliceExample:
    id: str
    path: str
    start: float | None
    end: float | None
    text: str


def accepted_ids(path: Path, marks: tuple[int, ...]) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"Marks file not found: {path}")
    accepted = set(marks)
    latest: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latest[row["ID"]] = int(row["mark"])
    return {label_id for label_id, score in latest.items() if score in accepted}


def load_slices(labels_path: Path, wanted: set[str] | None) -> list[SliceExample]:
    if not labels_path.is_file():
        raise SystemExit(f"Labels file not found: {labels_path}")
    found: dict[str, SliceExample] = {}
    present: set[str] = set()
    empty = 0
    with labels_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_id = row["ID"]
            present.add(label_id)
            if wanted is not None and label_id not in wanted:
                continue
            if label_id in found:
                continue
            text = row["text"].strip()
            if not text:
                empty += 1
                continue
            start, end = parse_timestamp(row["timestamp"])
            found[label_id] = SliceExample(label_id, row["path"], start, end, text)
    if empty:
        print(f"skipped {empty} accepted labels with empty text", flush=True)
    if wanted is not None:
        absent = [label_id for label_id in wanted if label_id not in present]
        if absent:
            raise SystemExit(f"{len(absent)} accepted ids are missing from {labels_path}, first: {absent[0]}")
    return list(found.values())


def cache_file(root: Path, slice_id: str) -> Path:
    # Two-character subfolders keep each directory at a few thousand files.
    return root / slice_id[:2] / f"{slice_id}.npy"


def write_cache_meta(root: Path, sample_rate: int, max_audio_seconds: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta = {"sample_rate": sample_rate, "max_audio_seconds": max_audio_seconds, "dtype": "int16"}
    path = root / CACHE_META
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != meta:
            raise SystemExit(f"{path} was written with {stored}, now {meta}. Use another student.voa_cache.")
        return
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def usable_cache(root: Path | None, sample_rate: int, max_audio_seconds: float) -> tuple[Path | None, str]:
    """The cache root if its slices fit these settings, else None and the reason."""
    if root is None:
        return None, "student.voa_cache is not set"
    path = root / CACHE_META
    if not path.is_file():
        return None, f"no {path}; run python main.py cache-voa"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("sample_rate") != sample_rate:
        return None, f"cache sample rate {meta.get('sample_rate')} != student.sample_rate {sample_rate}"
    if float(meta.get("max_audio_seconds", 0)) < max_audio_seconds:
        return None, f"cache slices are capped at {meta.get('max_audio_seconds')} s < {max_audio_seconds} s"
    return root, ""


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(audio * INT16_SCALE), -32768, 32767).astype(np.int16)


def read_cached(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32) / INT16_SCALE
