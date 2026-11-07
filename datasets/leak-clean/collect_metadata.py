"""Collect one metadata row per clip and check that every audio has metadata.

Only metadata and audio presence are read. Audio is never decoded.

- common_voice: split TSVs in ``paths.common_voice_splits``, audio files in
  ``paths.common_voice/clips``.
- fleurs: Arrow files listed in ``paths.fleurs/<train|validation|test>/state.json``.
- opentts: ``paths.opentts/voices/<voice>/train.jsonl`` and ``data/<voice>.zip``.
- tg_voices, ukr_dialects, cv10_test: ``paths.<name>/data/<split>-*.parquet``.
- voa: the pseudo-labels in ``student.labels`` give the slice texts.
  ``voa_prep.whisper_meta`` is checked against the audio in the shards
  under ``pseudo_labelling.voa``, and ``pseudo_labelling.timestamps``
  against the pseudo-labels.

Writes ``metadata.parquet`` (one row per clip that has audio and text) and
``coverage.csv`` (every audio/metadata check, including the ones that found
nothing) to ``output/leak-clean``. A dataset that is not on disk is reported
in ``coverage.csv`` and skipped.

    python datasets/leak-clean/collect_metadata.py
    python datasets/leak-clean/collect_metadata.py --datasets common_voice voa
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from leak_common import (
    AUDIO_SUFFIXES,
    CONFIG_PATH,
    COVERAGE_NAME,
    DATASET_ORDER,
    METADATA_NAME,
    config_file,
    config_text,
    configured_labelling_file,
    load_config,
    normalize_text,
    output_dir,
    repo_path,
)

COMMON_VOICE_SPLITS = {
    "train": "train_validated.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
    "unlabelled": "unlabelled.tsv",
}
# (folder or file prefix on disk, split name used everywhere else)
FLEURS_SPLITS = (("train", "train"), ("validation", "dev"), ("test", "test"))
PARQUET_DATASETS = {
    "tg_voices": ("transcription", (("train", "train"),)),
    "ukr_dialects": ("text", (("train", "train"), ("validation", "dev"), ("test", "test"))),
    "cv10_test": ("transcription", (("train", "test"),)),
}

# coverage.csv problem codes
NOT_FOUND = "dataset_not_found"
AUDIO_NOT_CHECKED = "audio_not_checked"
AUDIO_WITHOUT_METADATA = "audio_without_metadata"
METADATA_WITHOUT_AUDIO = "metadata_without_audio"
AUDIO_WITHOUT_TEXT = "audio_without_text"
DUPLICATE_AUDIO_NAME = "duplicate_audio_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Datasets to collect. Default: all.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: output/leak-clean.")
    return parser.parse_args()


class Collector:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.coverage: list[dict] = []

    def clip(self, dataset: str, split: str, clip_id: str, audio_name: str | None, location: str, text) -> bool:
        """Keep a clip that has text. Return False when its text is empty."""
        normalized = normalize_text(text)
        if not normalized:
            return False
        self.rows.append(
            {
                "dataset": dataset,
                "split": split,
                "id": str(clip_id),
                "audio_name": "" if not audio_name else Path(str(audio_name)).name,
                "location": location,
                "text": str(text).strip(),
                "text_norm": normalized,
                "words": len(normalized.split()),
            }
        )
        return True

    def problem(self, dataset: str, split: str, problem: str, examples: list[str], count: int | None = None) -> None:
        self.coverage.append(
            {
                "dataset": dataset,
                "split": split,
                "problem": problem,
                "count": len(examples) if count is None else count,
                "example": examples[0] if examples else "",
            }
        )


def audio_fields(column) -> tuple[list[int], list[str | None]]:
    """Byte size (0 when empty or null) and stored file name of each ``audio`` struct."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    names = [field.name for field in column.type]
    sizes = pc.fill_null(pc.binary_length(pc.struct_field(column, "bytes")), 0).to_pylist()
    paths = pc.struct_field(column, "path").to_pylist() if "path" in names else [None] * len(column)
    return sizes, paths


def common_voice(collector: Collector, config: dict) -> None:
    splits_dir = config_file(config, "paths.common_voice_splits")
    clips = config_file(config, "paths.common_voice") / "clips"
    on_disk: set[str] | None = None
    if clips.is_dir():
        on_disk = {entry.name for entry in os.scandir(clips) if entry.is_file()}
    else:
        collector.problem("common_voice", "", AUDIO_NOT_CHECKED, [str(clips)], count=1)

    listed: set[str] = set()
    for split, filename in COMMON_VOICE_SPLITS.items():
        path = splits_dir / filename
        if not path.is_file():
            collector.problem("common_voice", split, NOT_FOUND, [str(path)], count=1)
            continue
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["path", "sentence"],
            dtype=str,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            encoding="utf-8",
        )
        listed.update(frame["path"])
        missing: list[str] = []
        empty: list[str] = []
        for clip, sentence in zip(frame["path"], frame["sentence"]):
            if on_disk is not None and clip not in on_disk:
                missing.append(clip)
                continue
            if not collector.clip("common_voice", split, clip, clip, str(clips / clip), sentence):
                empty.append(clip)
        if on_disk is not None:
            collector.problem("common_voice", split, METADATA_WITHOUT_AUDIO, missing)
        collector.problem("common_voice", split, AUDIO_WITHOUT_TEXT, empty)

    if on_disk is not None:
        extra = sorted(name for name in on_disk - listed if Path(name).suffix.lower() in AUDIO_SUFFIXES)
        collector.problem("common_voice", "", AUDIO_WITHOUT_METADATA, extra)


def fleurs(collector: Collector, config: dict) -> None:
    root = config_file(config, "paths.fleurs")
    for folder, split in FLEURS_SPLITS:
        directory = root / folder
        state_path = directory / "state.json"
        if not state_path.is_file():
            collector.problem("fleurs", split, NOT_FOUND, [str(state_path)], count=1)
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        files = [directory / item["filename"] for item in state.get("_data_files", [])]
        missing = [str(path) for path in files if not path.is_file()]
        no_audio: list[str] = []
        empty: list[str] = []
        for path in files:
            if not path.is_file():
                continue
            for batch_index, batch in enumerate(ipc.open_stream(path)):
                texts = batch.column("transcription").to_pylist()
                identifiers = batch.column("id").to_pylist() if "id" in batch.schema.names else [None] * len(texts)
                sizes, names = audio_fields(batch.column("audio"))
                for offset, text in enumerate(texts):
                    location = f"{path}#batch={batch_index}&row={offset}"
                    # FLEURS ``id`` is the sentence, read by several speakers.
                    clip_id = f"{identifiers[offset]}/{names[offset] or offset}"
                    if sizes[offset] == 0:
                        no_audio.append(location)
                    elif not collector.clip("fleurs", split, clip_id, names[offset], location, text):
                        empty.append(location)
        collector.problem("fleurs", split, METADATA_WITHOUT_AUDIO, missing + no_audio)
        collector.problem("fleurs", split, AUDIO_WITHOUT_TEXT, empty)


def opentts(collector: Collector, config: dict) -> None:
    root = config_file(config, "paths.opentts")
    voices_dir = root / "voices"
    if not voices_dir.is_dir():
        collector.problem("opentts", "train", NOT_FOUND, [str(voices_dir)], count=1)
        return
    missing: list[str] = []
    extra: list[str] = []
    empty: list[str] = []
    collisions: list[str] = []
    for voice_dir in sorted(path for path in voices_dir.iterdir() if path.is_dir()):
        voice = voice_dir.name
        meta = voice_dir / "train.jsonl"
        archive = root / "data" / f"{voice}.zip"
        members: dict[str, zipfile.ZipInfo] = {}
        if archive.is_file():
            with zipfile.ZipFile(archive) as zipped:
                for info in zipped.infolist():
                    name = Path(info.filename).name
                    if info.is_dir() or Path(name).suffix.lower() not in AUDIO_SUFFIXES:
                        continue
                    if name in members:
                        collisions.append(f"{archive}!{info.filename}")
                    members[name] = info
        else:
            missing.append(str(archive))
        if not meta.is_file():
            extra.extend(f"{archive}!{info.filename}" for info in members.values())
            continue
        referenced: set[str] = set()
        for line_number, line in enumerate(meta.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = Path(str(row.get("audio") or "")).name
            info = members.get(name)
            if info is None or info.file_size == 0:
                missing.append(f"{meta}:{line_number}")
                continue
            referenced.add(name)
            location = f"{archive}!{info.filename}"
            if not collector.clip("opentts", "train", f"{voice}/{name}", name, location, row.get("text")):
                empty.append(location)
        extra.extend(f"{archive}!{info.filename}" for name, info in members.items() if name not in referenced)
    collector.problem("opentts", "train", METADATA_WITHOUT_AUDIO, missing)
    collector.problem("opentts", "train", AUDIO_WITHOUT_METADATA, extra)
    collector.problem("opentts", "train", AUDIO_WITHOUT_TEXT, empty)
    collector.problem("opentts", "train", DUPLICATE_AUDIO_NAME, collisions)


def parquet_dataset(collector: Collector, config: dict, dataset: str) -> None:
    text_column, layout = PARQUET_DATASETS[dataset]
    data_dir = config_file(config, f"paths.{dataset}") / "data"
    for prefix, split in layout:
        files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
        if not files:
            collector.problem(dataset, split, NOT_FOUND, [str(data_dir / f"{prefix}-*.parquet")], count=1)
            continue
        no_audio: list[str] = []
        empty: list[str] = []
        for path in files:
            parquet = pq.ParquetFile(path)
            available = parquet.schema_arrow.names
            if text_column not in available or "audio" not in available:
                raise SystemExit(f"{path} has columns {available}, expected {text_column} and audio")
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=[text_column, "audio"])
                texts = table.column(text_column).to_pylist()
                sizes, names = audio_fields(table.column("audio"))
                for offset, text in enumerate(texts):
                    location = f"{path}#rowgroup={group}&row={offset}"
                    clip_id = f"{path.name}:{group}:{offset}"
                    if sizes[offset] == 0:
                        no_audio.append(location)
                    elif not collector.clip(dataset, split, clip_id, names[offset], location, text):
                        empty.append(location)
        collector.problem(dataset, split, METADATA_WITHOUT_AUDIO, no_audio)
        collector.problem(dataset, split, AUDIO_WITHOUT_TEXT, empty)


def voa_labels_path(config: dict) -> Path:
    raw = config_text(config, "student.labels")
    if raw == "latest":
        return configured_labelling_file("labels.csv", config)
    return repo_path(raw)


def voa(collector: Collector, config: dict) -> None:
    # Audio against the clip metadata.
    shards = sorted((config_file(config, "pseudo_labelling.voa") / "data").glob("train-*.parquet"))
    meta_path = config_file(config, "voa_prep.whisper_meta")
    if not shards:
        collector.problem("voa", "", AUDIO_NOT_CHECKED, [str(config_file(config, "pseudo_labelling.voa") / "data")], count=1)
    elif not meta_path.is_file():
        collector.problem("voa", "", NOT_FOUND, [str(meta_path)], count=1)
    else:
        in_shards: set[str] = set()
        for shard in shards:
            in_shards.update(str(name) for name in pq.read_table(shard, columns=["audio.path"]).column("path").to_pylist())
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        records = payload["files"] if isinstance(payload, dict) else payload
        described = {str(record["path"]) for record in records if record.get("path")}
        collector.problem("voa", "", AUDIO_WITHOUT_METADATA, sorted(in_shards - described))
        collector.problem("voa", "", METADATA_WITHOUT_AUDIO, sorted(described - in_shards))

    # Slice texts: the pseudo-labels the student trains on.
    try:
        labels_path = voa_labels_path(config)
    except SystemExit as error:
        collector.problem("voa", "train", NOT_FOUND, [str(error)], count=1)
        return
    if not labels_path.is_file():
        collector.problem("voa", "train", NOT_FOUND, [str(labels_path)], count=1)
        return
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    empty: list[str] = []
    for label_id, audio_path, timestamp, text in zip(labels["ID"], labels["path"], labels["timestamp"], labels["text"]):
        location = f"{audio_path}@{timestamp or 'whole'}"
        if not collector.clip("voa", "train", label_id, audio_path, location, text):
            empty.append(label_id)

    # Slices that the time splits define but labelling never wrote also have no text.
    timestamps_path = config_file(config, "pseudo_labelling.timestamps")
    ukrainian_path = config_file(config, "pseudo_labelling.ukrainian_meta")
    if timestamps_path.is_file() and ukrainian_path.is_file():
        ukrainian = set(pd.read_csv(ukrainian_path, usecols=["path"], dtype=str)["path"])
        slices = pd.read_csv(timestamps_path, usecols=["ID", "path"], dtype=str)
        slices = slices[slices["path"].isin(ukrainian)]
        empty.extend(sorted(set(slices["ID"]) - set(labels["ID"])))
    collector.problem("voa", "train", AUDIO_WITHOUT_TEXT, empty)


def collect(datasets: list[str], destination: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write ``metadata.parquet`` and ``coverage.csv`` and return them."""
    config = load_config()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"config {CONFIG_PATH}", flush=True)

    collector = Collector()
    readers = {
        "common_voice": common_voice,
        "fleurs": fleurs,
        "opentts": opentts,
        "voa": voa,
    }
    for dataset in datasets:
        before = len(collector.rows)
        if dataset in PARQUET_DATASETS:
            parquet_dataset(collector, config, dataset)
        else:
            readers[dataset](collector, config)
        print(f"{dataset}: {len(collector.rows) - before} clips with text", flush=True)

    metadata = pd.DataFrame(
        collector.rows,
        columns=["dataset", "split", "id", "audio_name", "location", "text", "text_norm", "words"],
    )
    coverage = pd.DataFrame(collector.coverage, columns=["dataset", "split", "problem", "count", "example"])
    metadata.to_parquet(destination / METADATA_NAME, index=False)
    coverage.to_csv(destination / COVERAGE_NAME, index=False)

    problems = coverage[coverage["count"] > 0]
    if problems.empty:
        print("every audio has metadata and every metadata row has audio", flush=True)
    else:
        print("audio/metadata problems:", flush=True)
        for row in problems.itertuples(index=False):
            where = f"{row.dataset}/{row.split}" if row.split else row.dataset
            print(f"  {where}: {row.problem} {row.count} (e.g. {row.example})", flush=True)
    print(f"wrote {destination / METADATA_NAME} ({len(metadata)} clips)", flush=True)
    print(f"wrote {destination / COVERAGE_NAME}", flush=True)
    return metadata, coverage


def main() -> None:
    args = parse_args()
    collect(args.datasets, output_dir(args.output))


if __name__ == "__main__":
    main()
