"""Load the labelled speech corpora selected in ``config.yaml``.

Each configured split is a concatenation of dataset parts. Audio stays on
disk until a worker asks for one example: Common Voice is a file path,
FLEURS is an Arrow file, OpenTTS is read from the voice zips, and the
parquet corpora (TG Voices, Ukrainian Dialects, the Common Voice 10 test
set) decode one clip from its row group.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import av
import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch

from common_voice import load_split
from leak_filter import LEAK_FILTERED, apply_leak_filter
from settings import Settings, load_settings

# Config name -> which roles it can fill, and the directory name of that role.
from dataset_roles import ROLES  # noqa: E402


def decode_audio(source: str | bytes) -> tuple[torch.Tensor, int]:
    container = av.open(io.BytesIO(source)) if isinstance(source, bytes) else av.open(source)
    try:
        stream = container.streams.audio[0]
        sample_rate = int(stream.sample_rate)
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    finally:
        container.close()
    label = "audio bytes" if isinstance(source, bytes) else source
    if not chunks:
        raise RuntimeError(f"No audio frames in {label}")
    audio = np.concatenate(chunks, axis=1)
    audio = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(audio)), sample_rate


class _FramePart:
    def __init__(self, source: str, frame) -> None:
        self.source = source
        self.paths = frame["audio_path"].tolist()
        self.sentences = frame["sentence"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def example_id(self, index: int) -> str:
        return self.paths[index]

    def drop_leaks(self, leak_ids: set[str]) -> int:
        """Remove clips whose file name is in ``leak_ids``; returns how many went."""
        keep = [index for index, path in enumerate(self.paths) if Path(path).name not in leak_ids]
        removed = len(self.paths) - len(keep)
        self.paths = [self.paths[index] for index in keep]
        self.sentences = [self.sentences[index] for index in keep]
        return removed

    def read(self, index: int) -> tuple[str, torch.Tensor, int]:
        waveform, sample_rate = decode_audio(self.paths[index])
        return self.sentences[index], waveform, sample_rate


class _ArrowPart:
    def __init__(self, source: str, directory: Path, text_column: str) -> None:
        self.source = source
        self.text_column = text_column
        self.files = [str(path) for path in _arrow_files(directory)]
        self.rows = self._index()
        self._cache_key = None
        self._cache = None

    def _index(self) -> list[tuple[int, int, int, str, str]]:
        rows = []
        for file_index, path in enumerate(self.files):
            cursor = 0
            for batch_index, batch in enumerate(ipc.open_stream(path)):
                texts = batch.column(self.text_column).to_pylist()
                identifiers = batch.column("id").to_pylist() if "id" in batch.schema.names else None
                for offset, text in enumerate(texts):
                    sentence = "" if text is None else str(text).strip()
                    if sentence:
                        identifier = identifiers[offset] if identifiers is not None else cursor + offset
                        rows.append(
                            (
                                file_index,
                                batch_index,
                                offset,
                                sentence,
                                f"{self.source}/{identifier}",
                            )
                        )
                cursor += batch.num_rows
        return rows

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache_key"] = None
        state["_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def example_id(self, index: int) -> str:
        return self.rows[index][4]

    def _batch(self, file_index: int, batch_index: int):
        key = (file_index, batch_index)
        if self._cache_key != key:
            batch = None
            for index, candidate in enumerate(ipc.open_stream(self.files[file_index])):
                if index == batch_index:
                    batch = candidate
                    break
            if batch is None:
                raise RuntimeError(f"{self.source}: missing batch {batch_index} in {self.files[file_index]}")
            self._cache = batch.column("audio").to_pylist()
            self._cache_key = key
        return self._cache

    def read(self, index: int) -> tuple[str, torch.Tensor, int]:
        file_index, batch_index, offset, sentence, _example_id = self.rows[index]
        blob = self._batch(file_index, batch_index)[offset]["bytes"]
        if not blob:
            raise RuntimeError(f"{self.source}: empty audio at {self.example_id(index)}")
        waveform, sample_rate = decode_audio(blob)
        return sentence, waveform, sample_rate


class _ZipPart:
    def __init__(self, source: str, records: list[tuple[str, str, str, str]]) -> None:
        self.source = source
        self.records = records
        self._archive = None
        self._archive_path = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_path"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def example_id(self, index: int) -> str:
        return self.records[index][3]

    def read(self, index: int) -> tuple[str, torch.Tensor, int]:
        archive_path, member, sentence, _example_id = self.records[index]
        if self._archive is None or self._archive_path != archive_path:
            if self._archive is not None:
                self._archive.close()
            self._archive = zipfile.ZipFile(archive_path)
            self._archive_path = archive_path
        waveform, sample_rate = decode_audio(self._archive.read(member))
        return sentence, waveform, sample_rate


class _ParquetPart:
    def __init__(self, source: str, files: list[Path], text_column: str) -> None:
        self.source = source
        self.files = [str(path) for path in files]
        self.text_column = text_column
        self.rows = self._index()
        self._cache_key = None
        self._cache = None

    def _index(self) -> list[tuple[int, int, int, str, str]]:
        rows = []
        for file_index, path in enumerate(self.files):
            parquet = pq.ParquetFile(path)
            cursor = 0
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=[self.text_column])
                texts = table.column(self.text_column).to_pylist()
                for offset, text in enumerate(texts):
                    sentence = "" if text is None else str(text).strip()
                    if sentence:
                        rows.append(
                            (
                                file_index,
                                group,
                                offset,
                                sentence,
                                f"{self.source}/{file_index}:{cursor + offset}",
                            )
                        )
                cursor += len(texts)
        return rows

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache_key"] = None
        state["_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def example_id(self, index: int) -> str:
        return self.rows[index][4]

    def drop_leaks(self, leak_ids: set[str]) -> int:
        """Remove rows whose ``<file>:<row group>:<row>`` id is in ``leak_ids``; returns how many went."""
        names = [Path(path).name for path in self.files]
        keep = [row for row in self.rows if f"{names[row[0]]}:{row[1]}:{row[2]}" not in leak_ids]
        removed = len(self.rows) - len(keep)
        self.rows = keep
        return removed

    def _audio_bytes(self, file_index: int, group: int, offset: int) -> bytes:
        key = (file_index, group)
        if self._cache_key != key:
            table = pq.ParquetFile(self.files[file_index]).read_row_group(group, columns=["audio"])
            self._cache = table.column("audio").to_pylist()
            self._cache_key = key
        blob = self._cache[offset]["bytes"]
        if not blob:
            raise RuntimeError(f"{self.source}: empty audio at row group {group} offset {offset}")
        return blob

    def read(self, index: int) -> tuple[str, torch.Tensor, int]:
        file_index, group, offset, sentence, _example_id = self.rows[index]
        waveform, sample_rate = decode_audio(self._audio_bytes(file_index, group, offset))
        return sentence, waveform, sample_rate


class SpeechSplit:
    def __init__(self, parts: list, sources: tuple[str, ...]) -> None:
        self.parts = parts
        self.sources = sources
        self._offsets = []
        total = 0
        for part in parts:
            self._offsets.append(total)
            total += len(part)
        self._length = total
        self._selected: list[int] | None = None

    def __len__(self) -> int:
        if self._selected is None:
            return self._length
        return len(self._selected)

    def _locate(self, index: int) -> tuple[object, int]:
        if self._selected is not None:
            index = self._selected[index]
        part_index = 0
        while part_index + 1 < len(self._offsets) and self._offsets[part_index + 1] <= index:
            part_index += 1
        return self.parts[part_index], index - self._offsets[part_index]

    def example_id(self, index: int) -> str:
        part, local = self._locate(index)
        return part.example_id(local)

    def source_at(self, index: int) -> str:
        part, _local = self._locate(index)
        return part.source

    def read(self, index: int) -> tuple[str, torch.Tensor, int]:
        part, local = self._locate(index)
        return part.read(local)

    def take_stride(self, start: int, step: int) -> SpeechSplit:
        chosen = list(range(self._length) if self._selected is None else self._selected)
        shard = SpeechSplit(self.parts, self.sources)
        shard._offsets = self._offsets
        shard._length = self._length
        shard._selected = chosen[start::step]
        return shard


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"{label} is not a directory: {path}")
    return path


def _arrow_files(directory: Path) -> list[Path]:
    state_path = directory / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"FLEURS split not found: {directory}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    files = [directory / item["filename"] for item in state.get("_data_files", [])]
    missing = [path for path in files if not path.is_file()]
    if not files or missing:
        raise SystemExit(f"FLEURS split is incomplete: {directory}")
    return files


def _parquet_files(root: Path, folder_name: str) -> list[Path]:
    files = sorted((root / "data").glob(f"{folder_name}-*.parquet"))
    if not files:
        raise SystemExit(f"No {folder_name} parquet files in {root / 'data'}")
    return files


def _common_voice(settings: Settings, folder_name: str) -> _FramePart:
    clips = _require_dir(settings.common_voice, "paths.common_voice") / "clips"
    frame = load_split(folder_name, clips, settings.common_voice_splits)
    if frame.empty:
        raise SystemExit(f"common_voice {folder_name} has no clips")
    return _FramePart("common_voice", frame)


def _fleurs(settings: Settings, folder_name: str) -> _ArrowPart:
    root = _require_dir(settings.fleurs, "paths.fleurs")
    directory = root / folder_name
    part = _ArrowPart("fleurs", directory, "transcription")
    if len(part) == 0:
        raise SystemExit(f"FLEURS {folder_name} has no transcriptions")
    return part


def _opentts(settings: Settings) -> _ZipPart:
    root = _require_dir(settings.opentts, "paths.opentts")
    records = []
    voices = sorted(path.name for path in (root / "voices").iterdir() if path.is_dir())
    if not voices:
        raise SystemExit(f"No OpenTTS voices in {root / 'voices'}")
    for voice in voices:
        meta = root / "voices" / voice / "train.jsonl"
        archive = root / "data" / f"{voice}.zip"
        if not meta.is_file() or not archive.is_file():
            raise SystemExit(f"OpenTTS voice {voice} is missing {meta.name} or {archive.name}")
        with zipfile.ZipFile(archive) as zipped:
            members = {Path(name).name: name for name in zipped.namelist() if not name.endswith("/")}
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sentence = str(row.get("text") or "").strip()
            audio_name = Path(str(row.get("audio") or "")).name
            member = members.get(audio_name)
            if sentence and member:
                records.append((str(archive), member, sentence, f"opentts/{voice}/{Path(member).stem}"))
    if not records:
        raise SystemExit(f"OpenTTS has no labelled clips in {root}")
    return _ZipPart("opentts", records)


def _parquet(settings: Settings, source: str, folder_name: str) -> _ParquetPart:
    if source == "tg_voices":
        root = _require_dir(settings.tg_voices, "paths.tg_voices")
        text_column = "transcription"
    elif source == "ukr_dialects":
        root = _require_dir(settings.ukr_dialects, "paths.ukr_dialects")
        text_column = "text"
    elif source == "cv10_test":
        root = _require_dir(settings.cv10_test, "paths.cv10_test")
        text_column = "transcription"
    else:
        raise SystemExit(f"No parquet loader for {source}")
    part = _ParquetPart(source, _parquet_files(root, folder_name), text_column)
    if len(part) == 0:
        raise SystemExit(f"{source} {folder_name} has no transcriptions")
    return part


def _part(settings: Settings, name: str, role: str):
    if name not in ROLES:
        known = ", ".join(sorted(ROLES))
        raise SystemExit(f"data.{role}: unknown dataset '{name}'. Known names: {known}")
    if role not in ROLES[name]:
        available = ", ".join(sorted(ROLES[name]))
        raise SystemExit(f"data.{role}: '{name}' has no {role} split (available: {available})")
    folder_name = ROLES[name][role]
    if name == "common_voice":
        return _common_voice(settings, folder_name)
    if name == "fleurs":
        return _fleurs(settings, folder_name)
    if name == "opentts":
        return _opentts(settings)
    return _parquet(settings, name, folder_name)


def _train_part(settings: Settings, name: str):
    """A train part without the clips that also sit in dev or test (see ``leak_filter.py``)."""
    part = _part(settings, name, "train")
    if name in LEAK_FILTERED:
        part = apply_leak_filter(part, name)
        if len(part) == 0:
            raise SystemExit(f"{name} train has no clips left after the leak filter")
    return part


def load_train_split(names: tuple[str, ...], settings: Settings | None = None) -> SpeechSplit:
    """The train splits of ``names`` as one split, as ``data.train`` would load them."""
    cfg = settings or load_settings()
    return SpeechSplit([_train_part(cfg, name) for name in names], tuple(names))


def load_configured_splits(settings: Settings | None = None) -> dict[str, SpeechSplit]:
    cfg = settings or load_settings()
    selected = {"train": cfg.train_data, "dev": cfg.dev_data, "test": cfg.test_data}
    splits = {}
    for role, names in selected.items():
        parts = [_train_part(cfg, name) if role == "train" else _part(cfg, name, role) for name in names]
        splits[role] = SpeechSplit(parts, names)
    return splits
