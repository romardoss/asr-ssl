"""Collect VOA-ukr clip metadata into datasets/voa-ukr/voa_ukr_meta.json.

Same work as notebooks/voa_ukr_meta_collector.ipynb: for each train clip, store
container metadata, the original transcription, a Whisper medium transcript, and
the language Whisper detected. Whisper only hears the first 300 seconds.
Clips come from the local snapshot via src/voa_dataset.py.

The JSON file is both the checkpoint and the output. Paths that already
succeeded are skipped. A record with ``error`` is retried and replaced. The
file is rewritten after every clip.

Run from the repo root:

    python src/meta_collector.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voa_dataset import DATASET_ID, SPLIT, iter_clips, num_examples

META_PATH = Path(__file__).resolve().parents[2] / "datasets" / "voa-ukr" / "voa_ukr_meta.json"
ENV_PATH = REPO / ".env"
WHISPER_MODEL = "medium"
LANGUAGE_SECONDS = 120
_ENV_REF = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*):\+([^}]*)\}"
    r"|\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
    r"|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, separator, raw = line.partition("=")
        if not separator:
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        values[key.strip()] = raw
    return values


def expand_env(value: str, env: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        if match.group(1) is not None:
            if not env.get(match.group(1)):
                return ""
            return expand_env(match.group(2), env)
        return env.get(match.group(3) or match.group(4), "")

    return _ENV_REF.sub(replace, value)


def apply_dotenv() -> bool:
    """Load .env. The dynamic linker only reads LD_LIBRARY_PATH at startup, so the caller re-execs."""
    if os.environ.get("VOA_ENV_READY") == "1":
        return False
    raw = read_dotenv(ENV_PATH)
    if not raw:
        return False
    base = dict(os.environ)
    if not base.get("VIRTUAL_ENV") and (REPO / "env").is_dir():
        base["VIRTUAL_ENV"] = str(REPO / "env")
    for key, value in raw.items():
        expanded = expand_env(value, base)
        os.environ[key] = expanded
        base[key] = expanded
    os.environ["VOA_ENV_READY"] = "1"
    return True


def empty_meta() -> dict:
    return {
        "dataset": DATASET_ID,
        "split": SPLIT,
        "files": [],
    }


def load_meta(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"files": payload}
    payload.setdefault("dataset", DATASET_ID)
    payload.setdefault("split", SPLIT)
    payload.setdefault("files", [])
    return payload


def read_checkpoint(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        print(f"no checkpoint, starting empty ({path})")
        return empty_meta()
    try:
        data = load_meta(path)
    except json.JSONDecodeError:
        print(f"unreadable {path}, starting empty")
        return empty_meta()
    print(f"checkpoint files={len(data['files'])} path={path}")
    return data


def save_meta(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["n_files"] = len(data["files"])
    data["updated"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def labelled(text) -> bool:
    if text is None:
        return False
    return str(text).strip() not in {"", "-"}


def probe_audio(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate,format_name:stream=sample_rate,channels,codec_name,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(raw)
    stream = (info.get("streams") or [{}])[0]
    container = info.get("format") or {}

    def as_int(value):
        if value in (None, "", "N/A"):
            return None
        return int(float(value))

    def as_float(value):
        if value in (None, "", "N/A"):
            return None
        return float(value)

    return {
        "sample_rate": as_int(stream.get("sample_rate")),
        "num_channels": as_int(stream.get("channels")),
        "codec": stream.get("codec_name") or container.get("format_name"),
        "bit_rate": as_int(stream.get("bit_rate") or container.get("bit_rate")),
        "duration_seconds": as_float(container.get("duration")),
    }


def use_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_waveform(path: Path, sampling_rate: int = 16000, duration_seconds: int = LANGUAGE_SECONDS):
    import numpy as np

    # ffmpeg, not PyAV. PyAV's to_ndarray uses np.array(..., copy=False), which NumPy 2 rejects.
    # -t keeps decoding to the first LANGUAGE_SECONDS. Probe still uses the full file.
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            str(duration_seconds),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_whisper():
    from faster_whisper import WhisperModel

    device = "cuda" if use_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    print("whisper", device, compute_type)
    return model


def transcribe(model, path: Path):
    audio = load_waveform(path)
    seconds_per_segment = (
        model.feature_extractor.n_samples / model.feature_extractor.sampling_rate
    )
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        language_detection_segments=max(1, round(LANGUAGE_SECONDS / seconds_per_segment)),
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    probability = info.language_probability
    return text, info.language, None if probability is None else float(probability)


def collect(meta_path: Path = META_PATH) -> None:
    from tqdm.auto import tqdm

    meta = read_checkpoint(meta_path)
    done = set()
    retry_at = {}
    for position, item in enumerate(meta["files"]):
        item_path = item.get("path")
        if not item_path:
            continue
        if item.get("error"):
            retry_at[item_path] = position
        else:
            done.add(item_path)
    print("skipping", len(done), "retrying", len(retry_at))

    model = load_whisper()
    total = num_examples()
    for clip in tqdm(iter_clips(), total=total):
        index = clip["index"]
        audio = clip["audio"] or {}
        raw = audio.get("bytes")
        file_path = audio.get("path")
        if not file_path:
            file_path = hashlib.sha256(raw or b"").hexdigest()
        if file_path in done:
            continue

        original = clip.get("transcription")
        record = {
            "path": file_path,
            "stream_index": index,
            "split": SPLIT,
            "source": DATASET_ID,
            "speaker_id": None,
            "is_labelled": labelled(original),
            "transcription": original,
            "sample_rate": None,
            "duration_seconds": None,
            "num_channels": None,
            "codec": None,
            "bit_rate": None,
            "num_bytes": len(raw) if raw else None,
            "whisper_transcription": None,
            "language": None,
            "language_probability": None,
            "error": None,
        }
        clip_path = None
        try:
            if not raw:
                raise ValueError("audio bytes are missing")
            suffix = Path(file_path).suffix or ".mp3"
            clip_path = Path("/tmp/voa_current").with_suffix(suffix)
            clip_path.write_bytes(raw)
            record.update(probe_audio(clip_path))
            text, language, probability = transcribe(model, clip_path)
            record.update(
                {
                    "whisper_transcription": text,
                    "language": language,
                    "language_probability": probability,
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"error {file_path}: {record['error']}")
        finally:
            if clip_path is not None:
                clip_path.unlink(missing_ok=True)

        if file_path in retry_at:
            meta["files"][retry_at.pop(file_path)] = record
        else:
            meta["files"].append(record)
        done.add(file_path)
        save_meta(meta_path, meta)
        print(
            f"{len(done)} {file_path} "
            f"sr={record['sample_rate']} dur={record['duration_seconds']} "
            f"language={record['language']}"
        )

    print("wrote", meta_path, "files", len(meta["files"]))


if __name__ == "__main__":
    if apply_dotenv():
        os.execv(sys.executable, [sys.executable, *sys.argv])
    collect()
