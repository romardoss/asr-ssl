"""Split VOA metadata into Ukrainian-only and other-language CSVs.

A clip is Ukrainian-only when both meta files label it Ukrainian:
Whisper in ``voa_prep.whisper_meta`` (language == "uk") and MMS in
``voa_prep.mms_languages`` (argmax logit == "ukr"). Ukrainian clips go to
``pseudo_labelling.ukrainian_meta``, every other clip to ``voa_prep.other_languages``.

Run from the repo root:

    python src/split_language_meta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import config_file, load_config  # noqa: E402

CONFIG = load_config()
WHISPER_META_PATH = config_file(CONFIG, "voa_prep.whisper_meta")
LOGITS_PATH = config_file(CONFIG, "voa_prep.mms_languages")
UKRAINIAN_PATH = config_file(CONFIG, "pseudo_labelling.ukrainian_meta")
OTHER_PATH = config_file(CONFIG, "voa_prep.other_languages")

WHISPER_UKRAINIAN = "uk"
MMS_UKRAINIAN = "ukr"


def load_whisper_meta(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["files"] if isinstance(payload, dict) else payload
    table = pd.DataFrame(rows)
    if "path" not in table.columns:
        raise ValueError(f"{path} has no path column")
    table.insert(0, "ID", table["path"].map(lambda name: Path(str(name)).stem))
    return table


def load_mms_languages(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"logits not found: {path}")
    table = pd.read_csv(path)
    if "ID" not in table.columns:
        raise ValueError(f"{path} has no ID column")
    languages = [column for column in table.columns if column != "ID"]
    logits = table[languages].apply(pd.to_numeric, errors="coerce")
    valid = logits.notna().all(axis=1)
    predicted = logits.loc[valid].idxmax(axis=1)
    return pd.Series(predicted.to_numpy(), index=table.loc[valid, "ID"].astype(str), name="mms_language")


def split_meta(whisper: pd.DataFrame, mms_language: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = whisper.copy()
    table["mms_language"] = table["ID"].map(mms_language)
    ukrainian = table["language"].eq(WHISPER_UKRAINIAN) & table["mms_language"].eq(MMS_UKRAINIAN)
    return table.loc[ukrainian].copy(), table.loc[~ukrainian].copy()


def main() -> None:
    whisper = load_whisper_meta(WHISPER_META_PATH)
    ukrainian, other = split_meta(whisper, load_mms_languages(LOGITS_PATH))
    ukrainian.to_csv(UKRAINIAN_PATH, index=False)
    other.to_csv(OTHER_PATH, index=False)
    print(f"clips {len(whisper)}")
    print(f"ukrainian {len(ukrainian)} -> {UKRAINIAN_PATH}")
    print(f"other {len(other)} -> {OTHER_PATH}")


if __name__ == "__main__":
    main()
