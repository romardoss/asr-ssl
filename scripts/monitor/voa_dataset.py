"""Read the local robinhad/VOA-ukr snapshot.

The files live at ``/mnt/data/rberehun/voa-ukr`` (override with ``VOA_UKR_ROOT``).
Metadata collection and training should both import this module instead of
streaming the dataset from the Hub. Shard order matches that stream, so
``index`` is the same ``stream_index`` already stored in ``voa_ukr_meta.json``.

Each clip is::

    {
        "index": int,
        "audio": {"bytes": bytes, "path": str},
        "transcription": str,
    }

Audio stays compressed. Callers decode it themselves.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DATASET_ID = "robinhad/VOA-ukr"
SPLIT = "train"
DEFAULT_ROOT = Path("/mnt/data/rberehun/voa-ukr")


def resolve_root(root: Path | str | None = None) -> Path:
    if root is None:
        root = os.environ.get("VOA_UKR_ROOT", DEFAULT_ROOT)
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"VOA-ukr dataset not found: {path}")
    return path


def shards(root: Path | str | None = None, split: str = SPLIT) -> list[Path]:
    data_dir = resolve_root(root) / "data"
    found = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not found:
        raise FileNotFoundError(f"no {split} parquet shards in {data_dir}")
    return found


def num_examples(root: Path | str | None = None, split: str = SPLIT) -> int:
    """Count clips from parquet footers, without reading audio bytes."""
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in shards(root, split))


def iter_shard_clips(shard: Path):
    """Yield the clips in one parquet shard, without a global index."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(batch_size=1):
        for row in batch.to_pylist():
            audio = row.get("audio") or {}
            yield {
                "audio": {
                    "bytes": audio.get("bytes"),
                    "path": audio.get("path"),
                },
                "transcription": row.get("transcription"),
            }


def iter_clips(root: Path | str | None = None, split: str = SPLIT):
    """Yield clips one at a time, in shard order."""
    index = 0
    for path in shards(root, split):
        for clip in iter_shard_clips(path):
            clip["index"] = index
            index += 1
            yield clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the local VOA-ukr dataset.")
    parser.add_argument("--root", type=Path, default=None, help="dataset directory")
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="how many clips to print (path, transcription, byte size)",
    )
    args = parser.parse_args()

    root = resolve_root(args.root)
    files = shards(root, args.split)
    print(f"root {root}")
    print(f"split {args.split} shards {len(files)} clips {num_examples(root, args.split)}")
    for clip in iter_clips(root, args.split):
        if clip["index"] >= args.limit:
            break
        audio = clip["audio"]["bytes"] or b""
        text = clip["transcription"]
        preview = "" if text is None else str(text).replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "..."
        print(
            f"{clip['index']} {clip['audio']['path']} "
            f"bytes={len(audio)} transcription={preview!r}"
        )


if __name__ == "__main__":
    main()
