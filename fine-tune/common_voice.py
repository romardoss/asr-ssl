"""Load the local Ukrainian Common Voice splits.

Audio files live under ``$PATH_TO_COMMON_VOICE/clips``. Split membership comes
from the TSV files in ``meta/common-voice`` (train_validated, dev, test).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "meta" / "common-voice"
SPLIT_FILES = {
    "train": "train_validated.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
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
    load_dotenv()
    raw = os.environ.get("PATH_TO_COMMON_VOICE", "").strip()
    if not raw:
        raise SystemExit("PATH_TO_COMMON_VOICE is not set. Add it to .env or the environment.")
    root = Path(raw)
    clips = root / "clips"
    if not clips.is_dir():
        raise SystemExit(f"Common Voice clips directory not found: {clips}")
    return root


def whisper_model() -> str:
    load_dotenv()
    raw = os.environ.get("WHISPER_MODEL", "").strip()
    if not raw:
        raise SystemExit("WHISPER_MODEL is not set. Add it to .env or the environment.")
    return raw


def checkpoint_root() -> Path:
    load_dotenv()
    raw = os.environ.get("PATH_TO_CHECKPOINTS", "").strip()
    if not raw:
        raise SystemExit("PATH_TO_CHECKPOINTS is not set. Add it to .env or the environment.")
    return Path(raw)


def load_split(name: str, clips_dir: Path) -> pd.DataFrame:
    filename = SPLIT_FILES[name]
    path = META_DIR / filename
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    frame = pd.read_csv(path, sep="\t", usecols=["path", "sentence"], dtype=str)
    frame = frame.dropna(subset=["path", "sentence"])
    frame["sentence"] = frame["sentence"].str.strip()
    frame = frame[frame["sentence"].ne("")]
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
