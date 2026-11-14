"""Drop known data leakage from the train splits before any training.

``datasets/leak-clean/review_sha.ipynb`` and ``review_acoustic.ipynb`` write
``sha_matches.csv`` and ``acoustic_matches.csv`` to ``output/leak-clean`` (or to
``$LEAK_OUTPUT``). Each row is an eval clip (dev/test) whose recording is also
in train, with the train clip named by ``dataset_train``, ``split_train`` and
``id_train``. Those train clips are removed here. Dev and test are never touched:
they stay the reference.

The filter is mandatory for the datasets in ``LEAK_FILTERED``. When either CSV is
missing the run stops, because training on an unfiltered train would give
optimistic dev/test numbers.

``id_train`` is the id that ``collect_metadata.py`` wrote: the clip file name for
Common Voice and ``<parquet file>:<row group>:<row>`` for Ukrainian Dialects.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAK_FILTERED = ("common_voice", "ukr_dialects")
MATCH_FILES = ("sha_matches.csv", "acoustic_matches.csv")
REQUIRED_COLUMNS = ("dataset_train", "split_train", "id_train")


def leak_dir() -> Path:
    raw = os.environ.get("LEAK_OUTPUT")
    return Path(raw) if raw else ROOT / "output" / "leak-clean"


def load_leak_ids(dataset: str, folder: Path | None = None) -> dict[str, set[str]]:
    """Train ids of ``dataset`` listed in each match CSV, keyed by CSV name."""
    folder = folder or leak_dir()
    ids: dict[str, set[str]] = {}
    for name in MATCH_FILES:
        path = folder / name
        if not path.is_file():
            raise SystemExit(
                f"{path} not found. Leak filtering of the {dataset} train split is mandatory: run "
                "datasets/leak-clean/review_sha.ipynb and review_acoustic.ipynb and write their CSVs "
                "(or set LEAK_OUTPUT to the folder that has them)."
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SystemExit(f"{path} has no column(s) {', '.join(missing)}")
        rows = frame[(frame["dataset_train"] == dataset) & (frame["split_train"] == "train")]
        ids[name] = set(rows["id_train"])
    return ids


def apply_leak_filter(part, dataset: str):
    """Remove the listed leaked clips from a train ``part``; it must have ``drop_leaks``."""
    by_file = load_leak_ids(dataset)
    wanted = set().union(*by_file.values())
    size = len(part)
    removed = part.drop_leaks(wanted)
    listed = ", ".join(f"{name} {len(ids)}" for name, ids in by_file.items())
    print(f"leak filter {dataset} train: removed {removed} of {size} clips (listed: {listed})", flush=True)
    if removed < len(wanted):
        print(
            f"leak filter {dataset} train: WARNING {len(wanted) - removed} listed id(s) are not in this train split; "
            "the CSVs may come from different data",
            flush=True,
        )
    return part
