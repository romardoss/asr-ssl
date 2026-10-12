"""Run chunk.py once per VOA shard, with a cap on how many run at once.

Each process handles one parquet shard. When every shard has finished, the
per-shard progress files are merged into ``meta/languages/ukrainian_30s.csv``.

    python pseudo-labelling/vad-chunking/chunk_orchestrator.py --max-processes 6
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk

CHUNK_SCRIPT = Path(__file__).resolve().parent / "chunk.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-processes",
        type=int,
        required=True,
        help="How many shard processes may run at the same time.",
    )
    return parser.parse_args()


def merged_progress(shards: list[Path]) -> dict[str, list[tuple[float, float]]]:
    found: dict[str, list[tuple[float, float]]] = {}
    for shard in shards:
        found.update(chunk.load_progress(chunk.progress_path(shard)))
    return found


def stop(running: list[tuple[subprocess.Popen, Path, int]]) -> None:
    for process, _shard, _slot in running:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, _shard, _slot in running:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_shards(shards: list[Path], max_processes: int) -> None:
    pending = list(shards)
    running: list[tuple[subprocess.Popen, Path, int]] = []
    free_slots = list(range(1, max_processes + 1))
    with tqdm(total=len(shards), desc="shards", unit="shard", position=0, file=sys.stderr) as progress:
        while pending or running:
            while pending and len(running) < max_processes:
                shard = pending.pop(0)
                slot = free_slots.pop(0)
                environment = os.environ.copy()
                environment["CHUNK_BAR_POSITION"] = str(slot)
                process = subprocess.Popen(
                    [sys.executable, str(CHUNK_SCRIPT), "--shard", str(shard)],
                    start_new_session=True,
                    env=environment,
                )
                running.append((process, shard, slot))
            still_running: list[tuple[subprocess.Popen, Path, int]] = []
            failed: tuple[Path, int] | None = None
            for process, shard, slot in running:
                code = process.poll()
                if code is None:
                    still_running.append((process, shard, slot))
                elif code != 0:
                    failed = (shard, code)
                    free_slots.append(slot)
                else:
                    progress.update(1)
                    free_slots.append(slot)
            running = still_running
            if failed is not None:
                stop(running)
                shard, code = failed
                raise SystemExit(f"{shard.name} exited with status {code}")
            if pending and len(running) >= max_processes:
                time.sleep(0.5)


def main() -> None:
    args = parse_args()
    if args.max_processes < 1:
        raise SystemExit("--max-processes must be at least 1")
    shards = chunk.shard_paths(chunk.voa_root())
    frame = pd.read_csv(chunk.SOURCE_CSV)
    long_paths = set(frame.loc[frame["duration_seconds"] > chunk.MAX_SECONDS, "path"].astype(str))
    print(f"shards {len(shards)} long clips {len(long_paths)} max processes {args.max_processes}", flush=True)
    location = chunk.index_audio_paths(shards)
    absent = sorted(path for path in long_paths if path not in location)
    if absent:
        raise SystemExit(f"{len(absent)} audio paths are not in the VOA shards, first: {absent[0]}")

    run_shards(shards, args.max_processes)
    spans = merged_progress(shards)
    missing = sorted(path for path in long_paths if path not in spans)
    if missing:
        raise SystemExit(f"{len(missing)} long clips have no chunks, first: {missing[0]}")
    written = chunk.write_output(frame, spans, chunk.OUTPUT_CSV)
    print(f"wrote {written} rows -> {chunk.OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
