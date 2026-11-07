"""Remove dev and test clips from Common Voice train and write a cleaned test set.

``train_validated.tsv`` is Common Voice ``validated.tsv`` without
``test.tsv``. Its point is to train on every validated clip, which is much
more than the official ``train.tsv``. So only the clips themselves are
removed: a train row is dropped when its clip is in dev or test. Speakers and
sentences shared with dev and test stay in train on purpose.

Because of that, part of ``test.tsv`` is read by train speakers or repeats a
train sentence, and its WER is optimistic. ``test_cleaned.tsv`` keeps only
the test rows whose speaker (``client_id``) and normalized sentence are both
absent from train. It is not used in training or evaluation:
``score_common_voice_cleaned.py`` compares WER on it with the full test.
See ``datasets/common-voice/README.md``.

Rows are copied byte for byte. Running it again on a cleaned train removes
nothing and writes the same ``test_cleaned.tsv``.

    python datasets/leak-clean/clean_common_voice_train.py --dry-run
    python datasets/leak-clean/clean_common_voice_train.py
    python datasets/leak-clean/clean_common_voice_train.py --splits datasets/common-voice/raw
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "datasets" / "common-voice" / "raw"
TRAIN_NAME = "train_validated.tsv"
HELD_OUT_NAMES = ("dev.tsv", "test.tsv")
TEST_NAME = "test.tsv"
CLEANED_TEST_NAME = "test_cleaned.tsv"

# Same normalization as normalize_text in fine-tune/fine-tune.py, which WER uses.
_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Folder with the Common Voice TSVs (paths.common_voice_splits).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing.")
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


def read_lines(path: Path) -> tuple[bytes, list[bytes]]:
    lines = path.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines[0], lines[1:]


def read_with_lines(path: Path) -> tuple[pd.DataFrame, bytes, list[bytes]]:
    frame = read_split(path)
    header, rows = read_lines(path)
    if len(frame) != len(rows):
        raise SystemExit(f"{path}: parsed {len(frame)} rows, file has {len(rows)} lines")
    return frame, header, rows


def write_rows(path: Path, header: bytes, rows: list[bytes], keep: list[bool]) -> None:
    kept = [row for row, chosen in zip(rows, keep) if chosen]
    path.write_bytes(b"\n".join([header, *kept]) + b"\n")


def main() -> None:
    args = parse_args()
    splits = args.splits if args.splits.is_absolute() else ROOT / args.splits

    train_path = splits / TRAIN_NAME
    train, train_header, train_rows = read_with_lines(train_path)
    held_out = pd.concat([read_split(splits / name) for name in HELD_OUT_NAMES], ignore_index=True)
    drop = train["path"].isin(set(held_out["path"]))
    kept = train.loc[~drop]
    print(
        f"train rows {len(train)} -> {len(kept)} (removed {int(drop.sum())} dev/test clips); "
        f"speakers {train['client_id'].nunique()} -> {kept['client_id'].nunique()}"
    )

    test_path = splits / TEST_NAME
    test, test_header, test_rows = read_with_lines(test_path)
    speaker = test["client_id"].isin(set(kept["client_id"]))
    sentence = test["sentence"].map(normalize_text).isin(set(kept["sentence"].map(normalize_text)))
    clean = ~(speaker | sentence)
    print(
        f"test rows {len(test)}: train speaker {int(speaker.sum())}, train sentence {int(sentence.sum())}; "
        f"cleaned {int(clean.sum())} rows, {test.loc[clean, 'client_id'].nunique()} speakers"
    )

    if args.dry_run:
        print("dry run: files not changed")
        return
    if drop.any():
        write_rows(train_path, train_header, train_rows, (~drop).tolist())
        print(f"wrote {train_path}")
    else:
        print(f"{train_path}: nothing to remove")
    cleaned_path = splits / CLEANED_TEST_NAME
    write_rows(cleaned_path, test_header, test_rows, clean.tolist())
    print(f"wrote {cleaned_path}")


if __name__ == "__main__":
    main()
