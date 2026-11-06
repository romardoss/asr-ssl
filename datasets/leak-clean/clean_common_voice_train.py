"""Remove Common Voice train rows that leak into dev or test.

``train_validated.tsv`` was built as Common Voice ``validated.tsv`` without
``test.tsv``, so it still held every dev clip and shared sentences and
speakers with dev and test. This applies the rules of the official
``train.tsv``: a train row is dropped when its clip is in dev or test, when
its sentence is read in dev or test, or when its speaker (``client_id``)
recorded dev or test. Rows that stay are copied byte for byte.

Running it again on a cleaned file removes nothing.

    python datasets/leak-clean/clean_common_voice_train.py --dry-run
    python datasets/leak-clean/clean_common_voice_train.py
    python datasets/leak-clean/clean_common_voice_train.py --splits datasets/common-voice/raw
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "datasets" / "common-voice" / "raw"
TRAIN_NAME = "train_validated.tsv"
HELD_OUT_NAMES = ("dev.tsv", "test.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Folder with the Common Voice TSVs (paths.common_voice_splits).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without writing.")
    return parser.parse_args()


def read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Split file not found: {path}")
    # Same reading rules as fine-tune/common_voice.py: one clip per line, quotes are text.
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    splits = args.splits if args.splits.is_absolute() else ROOT / args.splits
    train_path = splits / TRAIN_NAME

    raw = train_path.read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    header, rows = lines[0], lines[1:]

    train = read_split(train_path)
    if len(train) != len(rows):
        raise SystemExit(f"{train_path}: parsed {len(train)} rows, file has {len(rows)} lines")
    held_out = pd.concat([read_split(splits / name) for name in HELD_OUT_NAMES], ignore_index=True)

    reasons = {
        "clip in dev/test": train["path"].isin(set(held_out["path"])),
        "sentence in dev/test": train["sentence"].isin(set(held_out["sentence"])),
        "speaker in dev/test": train["client_id"].isin(set(held_out["client_id"])),
    }
    drop = pd.Series(False, index=train.index)
    for reason, mask in reasons.items():
        drop |= mask
        print(f"{reason}: {int(mask.sum())} rows")
    kept = train.loc[~drop]
    print(
        f"train rows {len(train)} -> {len(kept)} (removed {int(drop.sum())}); "
        f"speakers {train['client_id'].nunique()} -> {kept['client_id'].nunique()}"
    )

    if not drop.any():
        print("nothing to remove")
        return
    if args.dry_run:
        print("dry run: file not changed")
        return
    kept_rows = [row for row, gone in zip(rows, drop.tolist()) if not gone]
    train_path.write_bytes(b"\n".join([header, *kept_rows]) + b"\n")
    print(f"wrote {train_path}")


if __name__ == "__main__":
    main()
