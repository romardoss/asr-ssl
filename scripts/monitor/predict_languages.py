"""Run facebook/mms-lid-256 language identification on the local VOA-ukr snapshot.

Audio stays in the parquet shards. Durations come from meta/voa_ukr_meta.json.
Each shard is packed into batches on its own, decoded in memory, then dropped.

Example:

    python src/predict_languages.py --output output/language_logits.csv --fp16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voa_dataset import iter_shard_clips, shards

DEFAULT_MODEL_ID = "facebook/mms-lid-256"
META_PATH = ROOT / "meta" / "voa_ukr_meta.json"
MAX_CLIP_SECONDS = 5 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the ID + per-language logit table to (CSV).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=META_PATH,
        help=f"Metadata JSON with duration_seconds (default: {META_PATH}).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="VOA-ukr dataset directory (default: VOA_UKR_ROOT or /mnt/data/rberehun/voa-ukr).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hub model id to load (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Max number of clips per batch (only meaningful when a GPU is available).",
    )
    parser.add_argument(
        "--max-batch-seconds",
        type=float,
        default=300.0,
        help=(
            "Max padded audio in one batch: longest clip times the clip count. "
            "Longer files get fewer batch-mates, which bounds peak GPU memory."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run inference in fp16 autocast on GPU (faster, slightly lower precision).",
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=None,
        help=(
            "If set, clips longer than this many seconds are not run through the model. "
            "Their logit cells are left empty."
        ),
    )
    return parser.parse_args()


def load_durations(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["files"] if isinstance(payload, dict) else payload
    durations = {}
    for row in rows:
        name = row.get("path")
        duration = row.get("duration_seconds")
        if name and duration is not None:
            durations[str(name)] = float(duration)
    return durations


def probe_duration(raw: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temporary:
        temporary.write(raw)
        temporary.flush()
        info = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                temporary.name,
            ],
            text=True,
        )
    duration = json.loads(info).get("format", {}).get("duration")
    if duration in (None, "", "N/A"):
        raise ValueError("ffprobe returned no duration")
    return float(duration)


def decode_mp3(raw: bytes, sampling_rate: int, max_seconds: float = MAX_CLIP_SECONDS) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temporary:
        temporary.write(raw)
        temporary.flush()
        pcm = subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                temporary.name,
                "-t",
                str(max_seconds),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(sampling_rate),
                "pipe:1",
            ]
        )
    return np.frombuffer(pcm, dtype=np.float32).copy()


def make_batches(
    paths_by_duration: list[str],
    max_items: int,
    max_seconds: float,
    durations: dict[str, float],
    max_clip_seconds: float = MAX_CLIP_SECONDS,
) -> list[list[str]]:
    """Greedily pack paths already sorted by duration.

    A batch is limited by item count and by longest_duration * count, which is
    the padded length the model actually sees. Clips longer than max_clip_seconds
    are packed as that length; decode keeps only the first max_clip_seconds.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_max_dur = 0.0
    for path in paths_by_duration:
        dur = min(durations[path], max_clip_seconds)
        candidate_max_dur = max(current_max_dur, dur)
        candidate_count = len(current) + 1
        if current and (
            candidate_count > max_items or candidate_max_dur * candidate_count > max_seconds
        ):
            batches.append(current)
            current = [path]
            current_max_dur = dur
        else:
            current.append(path)
            current_max_dur = candidate_max_dur
    if current:
        batches.append(current)
    return batches


def clip_id(path: str) -> str:
    return Path(path).stem


def load_existing(output_path: Path, lang_columns: list[str]) -> dict[str, list[float]]:
    if not output_path.is_file():
        return {}
    existing = pd.read_csv(output_path)
    if "ID" not in existing.columns or any(column not in existing.columns for column in lang_columns):
        print(f"ignoring {output_path}: columns do not match this model")
        return {}
    rows = {}
    for record in existing.to_dict(orient="records"):
        rows[str(record["ID"])] = [record[column] for column in lang_columns]
    print(f"resuming {len(rows)} rows from {output_path}")
    return rows


def write_table(rows: dict[str, list[float]], lang_columns: list[str], output_path: Path) -> None:
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=lang_columns)
    frame.index.name = "ID"
    frame = frame.sort_index().reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    args = parse_args()
    known_durations = load_durations(args.meta)
    shard_paths = shards(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size if device.type == "cuda" else 1
    print(f"shards {len(shard_paths)} device={device} batch_size={batch_size}")

    processor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_id).to(device).eval()
    sampling_rate = int(processor.sampling_rate)
    lang_columns = [model.config.id2label[i] for i in range(model.config.num_labels)]

    output_path = Path(args.output)
    rows = load_existing(output_path, lang_columns)
    done = set(rows)
    autocast_enabled = args.fp16 and device.type == "cuda"

    def run_batch(paths: list[str], audio_bytes: dict[str, bytes]) -> dict[str, list[float]]:
        waveforms = [decode_mp3(audio_bytes[path], sampling_rate) for path in paths]
        inputs = processor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=autocast_enabled):
                logits = model(**inputs).logits
            logits = logits.float().cpu()
            out = {clip_id(path): logits[i].tolist() for i, path in enumerate(paths)}
        except torch.cuda.OutOfMemoryError:
            print("CUDA OutOfMemoryError")
            del inputs
            torch.cuda.empty_cache()
            if len(paths) == 1:
                raise
            mid = len(paths) // 2
            out = run_batch(paths[:mid], audio_bytes)
            out.update(run_batch(paths[mid:], audio_bytes))
            return out
        del inputs, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    for shard in tqdm(shard_paths, desc="shards"):
        clips = list(iter_shard_clips(shard))
        durations: dict[str, float] = {}
        audio_bytes: dict[str, bytes] = {}
        skipped = False
        for clip in clips:
            audio = clip["audio"] or {}
            path = audio.get("path")
            raw = audio.get("bytes")
            if not path or not raw or clip_id(path) in done:
                continue
            try:
                duration = known_durations.get(path)
                if duration is None:
                    duration = probe_duration(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to read duration {path}: {exc}")
                continue
            if args.max_audio_length is not None and duration > args.max_audio_length:
                rows[clip_id(path)] = [None] * len(lang_columns)
                done.add(clip_id(path))
                skipped = True
                continue
            durations[path] = duration
            audio_bytes[path] = raw
        del clips

        if not durations:
            if skipped:
                write_table(rows, lang_columns, output_path)
            continue
        ordered = sorted(durations, key=durations.get)
        batches = make_batches(ordered, batch_size, args.max_batch_seconds, durations)
        print(f"{shard.name}: {len(durations)} clips, {len(batches)} batches")
        for paths in batches:
            rows.update(run_batch(paths, audio_bytes))
            done.update(clip_id(path) for path in paths)
        del audio_bytes
        write_table(rows, lang_columns, output_path)

    print(f"Wrote {len(rows)} rows x {len(lang_columns)} language columns to {output_path}")


if __name__ == "__main__":
    main()
