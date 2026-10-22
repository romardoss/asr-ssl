"""Split Ukrainian VOA clips so every row is at most 30 seconds.

Reads ``datasets/voa-ukr/ukrainian.csv``. A clip of 30 seconds or less is copied
unchanged. A longer clip is cut with Silero VAD: inside each 30 second window
the cut is the latest speech boundary that is still under 30 seconds. If that
window has no boundary, the cut is exactly 30 seconds. The next window starts
at that cut.

Audio bytes come from one VOA parquet shard (``--shard``). Progress for that
shard is appended to ``datasets/voa-ukr/vad-progress/<shard>.jsonl``, so several
shards can run at once. ``chunk_orchestrator.py`` starts those processes and
writes ``datasets/voa-ukr/ukrainian_30s.csv``.

    python pseudo-labelling/vad-chunking/chunk.py --shard /path/to/train-00000-of-00195.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT / "datasets" / "voa-ukr" / "ukrainian.csv"
OUTPUT_CSV = ROOT / "datasets" / "voa-ukr" / "ukrainian_30s.csv"
PROGRESS_DIR = ROOT / "datasets" / "voa-ukr" / "vad-progress"
DEFAULT_VOA_ROOT = Path("/mnt/data/rberehun/voa-ukr")
MAX_SECONDS = 30.0
VAD_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True, help="One VOA train parquet shard.")
    parser.add_argument("--source", type=Path, default=SOURCE_CSV)
    return parser.parse_args()


def progress_path(shard: Path) -> Path:
    return PROGRESS_DIR / f"{shard.stem}.jsonl"


def voa_root() -> Path:
    root = Path(os.environ.get("VOA_UKR_ROOT", DEFAULT_VOA_ROOT))
    data = root / "data"
    if not data.is_dir():
        raise SystemExit(f"VOA data directory not found: {data}")
    return root


def shard_paths(root: Path) -> list[Path]:
    found = sorted((root / "data").glob("train-*.parquet"))
    if not found:
        raise SystemExit(f"No train parquet shards in {root / 'data'}")
    return found


def paths_in_shard(shard: Path) -> set[str]:
    names = pq.read_table(shard, columns=["audio.path"]).column("path").to_pylist()
    return {str(name) for name in names}


def index_audio_paths(shards: list[Path]) -> dict[str, Path]:
    """Map an mp3 name to the shard that stores its bytes."""
    found: dict[str, Path] = {}
    for shard in shards:
        names = pq.read_table(shard, columns=["audio.path"]).column("path").to_pylist()
        for name in names:
            found[str(name)] = shard
    return found


def chunk_spans(duration: float, boundaries: list[float]) -> list[tuple[float, float]]:
    """Cut ``duration`` into pieces of at most 30 seconds.

    Each cut is the latest boundary strictly inside the current 30 second
    window. With no such boundary the cut is exactly 30 seconds.
    """
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor + MAX_SECONDS < duration:
        limit = cursor + MAX_SECONDS
        inside = [point for point in boundaries if cursor < point < limit]
        cut = max(inside) if inside else limit
        if cut <= cursor:
            cut = limit
        spans.append((cursor, cut))
        cursor = cut
    if cursor < duration:
        spans.append((cursor, duration))
    return spans


def speech_boundaries(waveform: np.ndarray, model) -> list[float]:
    import torch
    from silero_vad import get_speech_timestamps

    audio = torch.from_numpy(np.ascontiguousarray(waveform))
    segments = get_speech_timestamps(
        audio,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=True,
    )
    points = []
    for segment in segments:
        points.append(float(segment["start"]))
        points.append(float(segment["end"]))
    return sorted(set(points))


def decode_audio(audio_bytes: bytes) -> np.ndarray:
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
            str(VAD_SAMPLE_RATE),
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


def load_progress(path: Path) -> dict[str, list[tuple[float, float]]]:
    if not path.is_file():
        return {}
    done: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done[str(record["path"])] = [(float(start), float(end)) for start, end in record["spans"]]
    return done


def append_progress(path: Path, audio_path: str, spans: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"path": audio_path, "spans": spans}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def chunk_row(row: pd.Series, spans: list[tuple[float, float]] | None) -> list[dict]:
    base = row.to_dict()
    if spans is None:
        base["timestamp"] = ""
        return [base]
    pieces = []
    for index, (start, end) in enumerate(spans):
        piece = dict(base)
        piece["ID"] = f"{row['ID']}_{index:03d}"
        piece["duration_seconds"] = end - start
        piece["timestamp"] = f"{start:.3f}-{end:.3f}"
        piece["whisper_transcription"] = ""
        pieces.append(piece)
    return pieces


def spans_for_shard(shard: Path, needed: set[str], model) -> dict[str, list[tuple[float, float]]]:
    found: dict[str, list[tuple[float, float]]] = {}
    position = int(os.environ.get("CHUNK_BAR_POSITION", "0"))
    parquet = pq.ParquetFile(shard)
    with tqdm(
        total=len(needed),
        desc=shard.name,
        unit="clip",
        position=position,
        leave=False,
        file=sys.stderr,
    ) as progress:
        for batch in parquet.iter_batches(batch_size=1, columns=["audio"]):
            audio = batch.to_pylist()[0]["audio"]
            name = str(audio["path"])
            if name not in needed:
                continue
            waveform = decode_audio(audio["bytes"])
            duration = len(waveform) / VAD_SAMPLE_RATE
            boundaries = speech_boundaries(waveform, model)
            found[name] = chunk_spans(duration, boundaries)
            progress.update(1)
    return found


def write_output(frame: pd.DataFrame, spans_by_path: dict[str, list[tuple[float, float]]], output: Path) -> int:
    rows: list[dict] = []
    for _, row in frame.iterrows():
        duration = float(row["duration_seconds"])
        if duration <= MAX_SECONDS:
            rows.extend(chunk_row(row, None))
            continue
        spans = spans_by_path.get(str(row["path"]))
        if not spans:
            raise SystemExit(f"No VAD chunks for {row['path']}")
        rows.extend(chunk_row(row, spans))
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return len(table)


def main() -> None:
    args = parse_args()
    shard = args.shard
    if not shard.is_file():
        raise SystemExit(f"Shard not found: {shard}")
    if not args.source.is_file():
        raise SystemExit(f"Source CSV not found: {args.source}")
    frame = pd.read_csv(args.source, usecols=["path", "duration_seconds"])
    long_paths = set(frame.loc[frame["duration_seconds"] > MAX_SECONDS, "path"].astype(str))
    in_shard = paths_in_shard(shard)
    destination = progress_path(shard)
    done_paths = set(load_progress(destination))
    pending = (long_paths & in_shard) - done_paths
    print(
        f"{shard.name}: in shard {len(in_shard & long_paths)} already chunked {len(done_paths & long_paths)} pending {len(pending)}",
        flush=True,
    )
    if not pending:
        return

    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    model = load_silero_vad()
    finished = 0
    for audio_path, spans in spans_for_shard(shard, pending, model).items():
        append_progress(destination, audio_path, spans)
        finished += 1
    if finished != len(pending):
        raise SystemExit(f"{shard.name}: chunked {finished} of {len(pending)} clips")


if __name__ == "__main__":
    sys.exit(main())
