"""Find clips whose normalized text occurs in more than one clip.

Reads ``metadata.parquet`` written by ``collect_metadata.py``. Two clips are
the same when their normalized text matches (``leak_common.normalize_text``).
Only the train, dev, and test splits are compared; Common Voice unlabelled
stays out.

Writes to ``output/leak-clean``:

- ``duplicates.csv``: one row per clip whose text occurs more than once,
  with its group, where the clip is, and every dataset/split that has the text.
- ``units.csv``: clips per dataset/split, the base for the shares in the plots.

``kind`` of a group:

- ``within_split``: every copy is in one split of one dataset;
- ``between_splits``: copies in several splits of one dataset;
- ``between_datasets``: copies in more than one dataset.

    python datasets/leak-clean/find_overlaps.py
    python datasets/leak-clean/find_overlaps.py --min-words 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from leak_common import DUPLICATES_NAME, METADATA_NAME, SPLIT_ORDER, UNITS_NAME, output_dir, unit_key

KINDS = ("between_datasets", "between_splits", "within_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Compare only texts with at least this many words. Default: 1 (every text).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Folder with metadata.parquet. Default: output/leak-clean.")
    args = parser.parse_args()
    if args.min_words < 1:
        parser.error("--min-words must be at least 1")
    return args


def repeat_groups(frame: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clips whose ``key`` occurs more than once, and one row per repeated value.

    ``frame`` needs ``dataset``, ``unit``, ``id`` and ``key``. The clips get
    ``group``, ``kind``, ``copies`` and ``units`` (every dataset/split with this
    value) and come sorted by group. Groups are numbered from 1, between
    datasets first, then by copies.
    """
    copies = frame.groupby(key)["unit"].transform("size")
    repeated = frame[copies > 1].copy()
    if repeated.empty:
        group_columns = ["copies", "n_units", "n_datasets", "units", "kind", "kind_rank", "group"]
        for column in ("group", "kind", "copies", "units", "unit_rank"):
            repeated[column] = pd.Series(dtype=object)
        return repeated, pd.DataFrame(columns=group_columns)

    groups = repeated.groupby(key).agg(
        copies=("unit", "size"),
        n_units=("unit", "nunique"),
        n_datasets=("dataset", "nunique"),
        units=("unit", lambda values: "; ".join(sorted(set(values), key=unit_key))),
    )
    groups["kind"] = "within_split"
    groups.loc[groups["n_units"] > 1, "kind"] = "between_splits"
    groups.loc[groups["n_datasets"] > 1, "kind"] = "between_datasets"
    groups["kind_rank"] = groups["kind"].map(KINDS.index)
    groups = groups.sort_values(["kind_rank", "copies"], ascending=[True, False])
    groups["group"] = range(1, len(groups) + 1)

    repeated = repeated.join(groups[["group", "kind", "copies", "units"]], on=key)
    repeated["unit_rank"] = repeated["unit"].map(unit_key)
    repeated = repeated.sort_values(["group", "unit_rank", "id"])
    return repeated, groups


def find(folder: Path, min_words: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write ``duplicates.csv`` and ``units.csv`` from ``metadata.parquet`` and return them."""
    metadata_path = folder / METADATA_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"{metadata_path} not found. Run collect_metadata.py first.")

    metadata = pd.read_parquet(metadata_path)
    metadata = metadata[metadata["split"].isin(SPLIT_ORDER)].copy()
    metadata["unit"] = metadata["dataset"] + "/" + metadata["split"]

    units = metadata.groupby("unit").size().rename("clips").reset_index()
    units = units.iloc[sorted(range(len(units)), key=lambda index: unit_key(units["unit"].iloc[index]))]
    units.to_csv(folder / UNITS_NAME, index=False)

    compared = metadata[metadata["words"] >= min_words]
    duplicates, groups = repeat_groups(compared, "text_norm")
    columns = ["group", "kind", "text_norm", "words", "copies", "units", "dataset", "split", "id", "audio_name", "location", "text"]
    duplicates[columns].to_csv(folder / DUPLICATES_NAME, index=False)

    print(f"compared {len(compared)} clips with at least {min_words} word(s)")
    for kind in KINDS:
        chosen = groups[groups["kind"] == kind]
        print(f"{kind}: {len(chosen)} texts, {int(chosen['copies'].sum())} clips")

    train_texts = set(compared.loc[compared["split"] == "train", "text_norm"])
    for unit in units["unit"]:
        if unit.endswith("/train"):
            continue
        clips = compared[compared["unit"] == unit]
        leaked = int(clips["text_norm"].isin(train_texts).sum())
        total = int(units.loc[units["unit"] == unit, "clips"].iloc[0])
        print(f"{unit}: {leaked} of {total} clips have a text that is also in train ({100 * leaked / total:.1f}%)")
    print(f"wrote {folder / DUPLICATES_NAME} ({len(duplicates)} rows, {len(groups)} texts)")
    print(f"wrote {folder / UNITS_NAME}")
    return duplicates[columns], units


def main() -> None:
    args = parse_args()
    find(output_dir(args.output), args.min_words)


if __name__ == "__main__":
    main()
