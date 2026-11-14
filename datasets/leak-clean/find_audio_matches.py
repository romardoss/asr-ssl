"""Find the same audio in two places: by file name and by SHA-256 of its bytes.

Reads ``metadata.parquet`` written by ``collect_metadata.py``. Only the train,
dev, and test splits are compared, as in ``find_overlaps.py``. Text plays no
part here: the same recording can carry a differently written transcript.

Two keys:

- ``name``: ``audio_name``, the file name stored with the clip. Common Voice
  and ``cv10_test`` keep the original ``common_voice_uk_<number>.mp3``, which
  stays the same across Common Voice versions. A name that repeats inside one
  dataset/split (``audio.wav`` in every row, say) does not identify a clip and
  is ignored everywhere. Different datasets can still number files the same
  way by chance, so a clip is kept only when a clip with the same name in
  another dataset/split has a similar text (``--min-text-similarity``) or the
  same bytes; ``same_bytes`` then shows whether all copies share their bytes.
- ``sha256``: ``audio_sha256``, the SHA-256 of the encoded audio bytes. It
  finds copies, renamed files, and one clip stored in two datasets. A
  re-encoded file has other bytes and is not found. VOA slices and clips
  collected with ``--no-audio-hash`` have no hash.

Writes ``audio_matches.csv`` to ``output/leak-clean``: one row per clip whose
name or hash occurs more than once, with ``key``, its group, kind
(``within_split``, ``between_splits``, ``between_datasets``), and every
dataset/split that has it.

    python datasets/leak-clean/find_audio_matches.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from find_overlaps import KINDS, repeat_groups
from leak_common import AUDIO_MATCHES_NAME, METADATA_NAME, SPLIT_ORDER, output_dir, unit_key

KEYS = {"name": "audio_name", "sha256": "audio_sha256"}
# Share of matching characters (difflib ratio of the normalized texts) from which two clips with
# the same file name count as the same recording. Unrelated sentences score about 0.2-0.4.
MIN_TEXT_SIMILARITY = 0.6
COLUMNS = [
    "key",
    "group",
    "kind",
    "value",
    "copies",
    "units",
    "same_bytes",
    "dataset",
    "split",
    "id",
    "audio_name",
    "audio_sha256",
    "location",
    "text_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Folder with metadata.parquet. Default: output/leak-clean.")
    parser.add_argument(
        "--min-text-similarity",
        type=float,
        default=MIN_TEXT_SIMILARITY,
        help=(
            "A match by file name counts only if the texts of two copies are at least this similar "
            f"(0 to 1, difflib ratio of normalized texts). 0 turns the check off. Default: {MIN_TEXT_SIMILARITY}."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.min_text_similarity <= 1:
        parser.error("--min-text-similarity must be between 0 and 1")
    return args


def load_compared(folder: Path) -> pd.DataFrame:
    metadata_path = folder / METADATA_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"{metadata_path} not found. Run collect_metadata.py first.")
    metadata = pd.read_parquet(metadata_path)
    if "audio_sha256" not in metadata.columns:  # written before hashing existed
        metadata["audio_sha256"] = ""
    for column in ("audio_name", "audio_sha256"):
        metadata[column] = metadata[column].fillna("").astype(str)
    metadata = metadata[metadata["split"].isin(SPLIT_ORDER)].copy()
    metadata["unit"] = metadata["dataset"] + "/" + metadata["split"]
    return metadata


def same_bytes(hashes: pd.Series) -> str:
    """``yes`` when every copy has one hash, ``no`` when hashes differ, else ``unknown``."""
    known = set(hashes[hashes != ""])
    if len(known) > 1:
        return "no"
    if len(known) == 1 and (hashes != "").all():
        return "yes"
    return "unknown"


def similar_text(first: str, second: str, threshold: float) -> bool:
    """True when the normalized texts are alike, or when one is missing (nothing to compare)."""
    if not first or not second:
        return True
    matcher = SequenceMatcher(None, first, second, autojunk=False)
    return matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold


def drop_unrelated_names(chosen: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    """Keep a clip only if a clip with its name in another dataset/split has a similar text or the same bytes.

    Returns the kept clips and how many clips that share their name with another
    one were dropped because every such copy has an unrelated text.
    """
    shared = chosen[chosen.duplicated("audio_name", keep=False)]
    if threshold <= 0 or shared.empty:
        return chosen, 0
    kept = set()
    for _, group in shared.groupby("audio_name"):
        rows = list(group[["unit", "text_norm", "audio_sha256"]].itertuples())
        for first in rows:
            for second in rows:
                if first.Index == second.Index or first.unit == second.unit:
                    continue
                same_audio = first.audio_sha256 != "" and first.audio_sha256 == second.audio_sha256
                if same_audio or similar_text(first.text_norm, second.text_norm, threshold):
                    kept.add(first.Index)
                    break
    dropped = shared.index.difference(list(kept))
    return chosen.drop(index=dropped), len(dropped)


def match(metadata: pd.DataFrame, min_text_similarity: float = MIN_TEXT_SIMILARITY) -> tuple[pd.DataFrame, list[str]]:
    """Clips whose name or hash repeats, and the names ignored as not unique in their split."""
    named = metadata[metadata["audio_name"] != ""]
    per_unit = named.groupby(["unit", "audio_name"]).size()
    ignored = sorted(set(per_unit[per_unit > 1].index.get_level_values("audio_name")))

    frames = []
    for key, column in KEYS.items():
        chosen = metadata[metadata[column] != ""]
        if key == "name":
            chosen = chosen[~chosen["audio_name"].isin(ignored)]
            chosen, dropped = drop_unrelated_names(chosen, min_text_similarity)
            if dropped:
                print(f"dropped {dropped} clips whose name repeats but whose text is unrelated (similarity < {min_text_similarity})")
        repeated, _ = repeat_groups(chosen, column)
        if repeated.empty:
            continue
        repeated["key"] = key
        repeated["value"] = repeated[column]
        if key == "name":
            repeated["same_bytes"] = repeated.groupby("group")["audio_sha256"].transform(same_bytes)
        else:
            repeated["same_bytes"] = "yes"
        frames.append(repeated)
    if not frames:
        return pd.DataFrame(columns=COLUMNS), ignored
    return pd.concat(frames, ignore_index=True)[COLUMNS], ignored


def eval_audio_leaks(matches: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per dev or test split: clips whose name or hash is also in some train split."""
    units = sorted(set(metadata["unit"]), key=unit_key)
    copies = matches[matches["split"] != "train"].copy()
    copies["unit"] = copies["dataset"] + "/" + copies["split"]
    copies["trains"] = copies["units"].map(lambda listed: [unit for unit in listed.split("; ") if unit.endswith("/train")])
    copies = copies[copies["trains"].map(bool).astype(bool)]
    rows = []
    for unit in units:
        if unit.endswith("/train"):
            continue
        clips = metadata[metadata["unit"] == unit]
        chosen = copies[copies["unit"] == unit]
        by_name = set(chosen.loc[chosen["key"] == "name", "id"])
        by_name_bytes = set(chosen.loc[(chosen["key"] == "name") & (chosen["same_bytes"] == "yes"), "id"])
        by_hash = set(chosen.loc[chosen["key"] == "sha256", "id"])
        # A clip can match by name and by hash; count each train split once per clip.
        per_clip = chosen.groupby("id")["trains"].agg(lambda lists: {train for trains in lists for train in trains})
        sources = Counter(train for trains in per_clip for train in trains)
        rows.append(
            {
                "unit": unit,
                "clips": len(clips),
                "hashed": int((clips["audio_sha256"] != "").sum()),
                "named": int((clips["audio_name"] != "").sum()),
                "name": len(by_name),
                "name_same_bytes": len(by_name_bytes),
                "sha256": len(by_hash),
                "either": len(by_name | by_hash),
                "sources": "; ".join(f"{train} {count}" for train, count in sorted(sources.items(), key=lambda item: -item[1])),
            }
        )
    columns = ["unit", "clips", "hashed", "named", "name", "name_same_bytes", "sha256", "either", "sources"]
    return pd.DataFrame(rows, columns=columns)


def find_audio(folder: Path, min_text_similarity: float = MIN_TEXT_SIMILARITY) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Write ``audio_matches.csv`` and return it, the dev/test summary, and the ignored names."""
    metadata = load_compared(folder)
    matches, ignored = match(metadata, min_text_similarity)
    matches.to_csv(folder / AUDIO_MATCHES_NAME, index=False)
    leaks = eval_audio_leaks(matches, metadata)

    hashed = int((metadata["audio_sha256"] != "").sum())
    print(f"compared {len(metadata)} clips, {hashed} with audio SHA-256")
    if ignored:
        print(f"ignored {len(ignored)} file names that repeat inside one split (e.g. {ignored[0]})")
    for key in KEYS:
        groups = matches[matches["key"] == key].drop_duplicates("group")
        counts = ", ".join(f"{kind} {int((groups['kind'] == kind).sum())}" for kind in KINDS)
        print(f"{key}: {counts}")
    for row in leaks.itertuples(index=False):
        print(
            f"{row.unit}: {row.either} of {row.clips} clips have the same audio in train "
            f"(name {row.name}, of them same bytes {row.name_same_bytes}; sha256 {row.sha256} of {row.hashed} hashed)"
        )
    print(f"wrote {folder / AUDIO_MATCHES_NAME} ({len(matches)} rows)")
    return matches, leaks, ignored


def main() -> None:
    args = parse_args()
    find_audio(output_dir(args.output), args.min_text_similarity)


if __name__ == "__main__":
    main()
