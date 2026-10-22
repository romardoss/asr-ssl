"""Turn VOA clips into slices of at most ``data.max_audio_seconds``.

A path is kept only when it is listed in ``pseudo_labelling.ukrainian_meta``.
Cuts come from ``pseudo_labelling.timestamps``. An empty timestamp means the
whole clip. Audio is decoded and resampled to ``pseudo_labelling.sample_rate``.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from settings import LabellingSettings


@dataclass(frozen=True)
class Segment:
    id: str
    path: str
    start: float | None
    end: float | None


def ukrainian_paths(settings: LabellingSettings) -> set[str]:
    if not settings.ukrainian_meta.is_file():
        raise SystemExit(f"Ukrainian metadata not found: {settings.ukrainian_meta}")
    frame = pd.read_csv(settings.ukrainian_meta, usecols=["path"])
    return set(frame["path"].astype(str))


def load_segments(settings: LabellingSettings, allowed: set[str]) -> list[Segment]:
    if not settings.timestamps.is_file():
        raise SystemExit(f"Timestamp metadata not found: {settings.timestamps}")
    frame = pd.read_csv(settings.timestamps)
    required = {"path", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"{settings.timestamps} is missing columns: {', '.join(sorted(missing))}")
    segments: list[Segment] = []
    seen: dict[str, int] = defaultdict(int)
    for row in frame.itertuples(index=False):
        audio_path = str(row.path)
        if audio_path not in allowed:
            continue
        start, end = parse_timestamp(getattr(row, "timestamp"))
        if "ID" in frame.columns and str(getattr(row, "ID")).strip():
            segment_id = str(row.ID)
        else:
            segment_id = f"{Path(audio_path).stem}_{seen[audio_path]:03d}"
            seen[audio_path] += 1
        segments.append(Segment(segment_id, audio_path, start, end))
    return segments


def parse_timestamp(value) -> tuple[float | None, float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, None
    start_text, end_text = text.split("-", 1)
    return float(start_text), float(end_text)


def voa_shards(settings: LabellingSettings) -> list[Path]:
    data = settings.voa / "data"
    found = sorted(data.glob("train-*.parquet"))
    if not found:
        raise SystemExit(f"No train parquet shards in {data}")
    return found


def index_shards(shards: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for shard in shards:
        names = pq.read_table(shard, columns=["audio.path"]).column("path").to_pylist()
        for name in names:
            found[str(name)] = shard
    return found


def decode(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(message or "ffmpeg failed to decode audio")
    if not result.stdout:
        raise RuntimeError("ffmpeg returned no audio")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def slice_audio(waveform: np.ndarray, segment: Segment, sample_rate: int, max_seconds: float) -> np.ndarray:
    if segment.start is None or segment.end is None:
        audio = waveform
    else:
        start = max(0, int(round(segment.start * sample_rate)))
        end = min(len(waveform), int(round(segment.end * sample_rate)))
        audio = waveform[start:end]
    limit = int(round(max_seconds * sample_rate))
    if len(audio) > limit:
        audio = audio[:limit]
    return np.ascontiguousarray(audio)


def iter_shard_audio(shard: Path, needed: set[str]):
    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(batch_size=1, columns=["audio"]):
        audio = batch.to_pylist()[0]["audio"]
        name = str(audio["path"])
        if name in needed:
            yield name, audio["bytes"]
