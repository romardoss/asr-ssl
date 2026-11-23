"""Decode the student's VOA slices once into ``student.voa_cache``.

Each slice becomes ``<voa_cache>/<ID[:2]>/<ID>.npy``: mono int16 PCM at
``student.sample_rate``, cut by its timestamp and capped at
``student.max_audio_seconds``, the same audio the student gets from ffmpeg.
The student then reads these files instead of running ffmpeg per mp3, which
frees the CPU (two runs at once were starving both GPUs).

By default the slices are those the student trains on: ``student.labels``
filtered by ``student.marks`` and ``student.accepted_mark``. ``--all`` caches
every labelled slice (any mark), so changing ``accepted_mark`` needs no new
pass. Already cached slices are skipped, so an interrupted run just restarts.

    python main.py cache-voa [--all] [--workers N]
    python fine-tune/cache_voa.py [--all] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
ROOT = FINE_TUNE_DIR.parent
# fine-tune first: pseudo-labelling has its own settings.py.
sys.path.insert(0, str(ROOT / "pseudo-labelling"))
sys.path.insert(0, str(FINE_TUNE_DIR))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from preprocess import Segment, decode, index_shards, iter_shard_audio, slice_audio  # noqa: E402
from settings import load_student_settings  # noqa: E402
from voa_slices import (  # noqa: E402
    SliceExample,
    accepted_ids,
    cache_file,
    load_slices,
    to_int16,
    write_cache_meta,
)

# Set in each pool worker by _init.
_ROOT: Path
_SAMPLE_RATE: int
_MAX_SECONDS: float


def _init(root: Path, sample_rate: int, max_seconds: float) -> None:
    global _ROOT, _SAMPLE_RATE, _MAX_SECONDS
    _ROOT, _SAMPLE_RATE, _MAX_SECONDS = root, sample_rate, max_seconds


def _save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and renamed, so a killed run never leaves a half file under the final name.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, to_int16(audio))
    temporary.replace(path)


def _cache_shard(job: tuple[Path, dict[str, list[SliceExample]]]) -> tuple[int, int, list[str]]:
    """Decode every needed file of one parquet shard once and save its slices."""
    shard, by_path = job
    written = 0
    seconds = 0.0
    errors: list[str] = []
    for name, data in iter_shard_audio(shard, set(by_path)):
        try:
            wave = decode(data, _SAMPLE_RATE)
        except RuntimeError as error:
            errors.append(f"{name}: {error}")
            continue
        for example in by_path[name]:
            audio = slice_audio(
                wave, Segment(example.id, example.path, example.start, example.end), _SAMPLE_RATE, _MAX_SECONDS
            )
            if audio.size == 0:
                errors.append(f"{example.id}: empty slice")
                continue
            _save(cache_file(_ROOT, example.id), audio)
            written += 1
            seconds += audio.size / _SAMPLE_RATE
    return written, int(seconds), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Cache every labelled slice, not only accepted marks.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel shards (ffmpeg processes).")
    args = parser.parse_args()

    cfg = load_student_settings()
    if cfg.voa_cache is None:
        raise SystemExit("Set student.voa_cache in the config first.")
    root = cfg.voa_cache
    write_cache_meta(root, cfg.sample_rate, cfg.max_audio_seconds)

    if cfg.filter_marks and not args.all:
        examples = load_slices(cfg.labels, accepted_ids(cfg.marks, cfg.accepted_mark))
        scope = f"marks {list(cfg.accepted_mark)}"
    else:
        examples = load_slices(cfg.labels, None)
        scope = "all marks"
    todo = [example for example in examples if not cache_file(root, example.id).is_file()]
    print(f"{cfg.labels}\n{scope}: {len(examples)} slices, {len(examples) - len(todo)} cached, {len(todo)} to do -> {root}")
    if not todo:
        return

    shards = sorted((cfg.voa / "data").glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"No train parquet shards in {cfg.voa / 'data'}")
    location = index_shards(shards, root / "shard_index.json")
    jobs: dict[Path, dict[str, list[SliceExample]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    for example in todo:
        shard = location.get(example.path)
        if shard is None:
            missing += 1
            continue
        jobs[shard][example.path].append(example)
    if missing:
        print(f"warning: {missing} slices point to audio absent from the VOA shards", file=sys.stderr)

    written = 0
    seconds = 0
    errors: list[str] = []
    # Biggest shards first, so the last workers are not left with a long one.
    ordered = sorted(jobs.items(), key=lambda item: -sum(len(slices) for slices in item[1].values()))
    with Pool(min(args.workers, len(ordered)), initializer=_init, initargs=(root, cfg.sample_rate, cfg.max_audio_seconds)) as pool:
        progress = tqdm(total=len(todo), desc="cache slices", unit="slice", file=sys.stderr)
        for count, duration, failed in pool.imap_unordered(_cache_shard, ordered):
            written += count
            seconds += duration
            errors.extend(failed)
            progress.update(count + len(failed))
        progress.close()

    print(f"written {written} slices, {seconds / 3600:.1f} h, about {seconds * cfg.sample_rate * 2 / 1e9:.1f} GB")
    if errors:
        log = root / "errors.txt"
        log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} failures, listed in {log}; those slices are decoded from mp3 during training")


if __name__ == "__main__":
    main()
