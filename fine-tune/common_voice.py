"""Load the local Ukrainian Common Voice splits.

Audio files live under ``paths.common_voice`` / ``clips`` from ``config.yaml``.
Split membership comes from the TSV files in ``datasets/common-voice/raw``
(train_validated, dev, test, unlabelled).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

from settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "datasets" / "common-voice"
RAW_DIR = META_DIR / "raw"
SPLIT_FILES = {
    "train": "train_validated.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
    "unlabelled": "unlabelled.tsv",
}


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing vars."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip()))


def common_voice_root() -> Path:
    root = load_settings().common_voice
    clips = root / "clips"
    if not clips.is_dir():
        raise SystemExit(f"Common Voice clips directory not found: {clips}")
    return root


def whisper_model() -> str:
    return load_settings().model_id


def checkpoint_root() -> Path:
    return load_settings().checkpoints


def load_split(name: str, clips_dir: Path) -> pd.DataFrame:
    filename = SPLIT_FILES[name]
    path = RAW_DIR / filename
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    # Common Voice TSVs are one clip per line. Quotes inside a sentence are
    # text, so the default CSV quote rules must stay off or one unmatched
    # quote swallows the following rows.
    source_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["path", "sentence"],
        dtype=str,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    if len(frame) != source_rows:
        raise SystemExit(f"{name}: parsed {len(frame)} rows, TSV has {source_rows}")
    frame = frame.dropna(subset=["path", "sentence"])
    frame["sentence"] = frame["sentence"].str.strip()
    frame = frame[frame["sentence"].ne("")]
    merged = frame["sentence"].str.contains(r"[\t\n]|common_voice_uk_", regex=True)
    if merged.any():
        raise SystemExit(f"{name}: {int(merged.sum())} sentence(s) still contain another row")
    frame["audio_path"] = frame["path"].map(lambda name: str(clips_dir / name))
    frame["split"] = name
    present = frame["audio_path"].map(lambda item: Path(item).is_file())
    missing = int((~present).sum())
    if missing:
        print(f"{name}: dropping {missing} row(s) whose audio file is missing")
    frame = frame.loc[present, ["audio_path", "sentence", "split"]].reset_index(drop=True)
    return frame


def load_splits() -> dict[str, pd.DataFrame]:
    clips_dir = common_voice_root() / "clips"
    return {name: load_split(name, clips_dir) for name in SPLIT_FILES}
