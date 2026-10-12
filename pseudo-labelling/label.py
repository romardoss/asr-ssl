"""Label Ukrainian VOA slices with the teacher model from ``config.yaml``.

Clips that are not listed in ``pseudo_labelling.ukrainian_meta`` are skipped.
Each remaining clip is resampled and cut on ``pseudo_labelling.timestamps``.
Transcripts are appended to ``labels.csv`` under
``pseudo_labelling.output_dir/<teacher>-XX``. With ``resume: true``, the newest
unfinished folder for that teacher is continued.

    python pseudo-labelling/label.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from preprocess import (
    Segment,
    decode,
    index_shards,
    iter_shard_audio,
    load_segments,
    slice_audio,
    ukrainian_paths,
    voa_shards,
)
from settings import LabellingSettings, load_labelling

LABELS_NAME = "labels.csv"
COMPLETE_NAME = "complete"


def pin_gpus(settings: LabellingSettings) -> None:
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in settings.gpus)


def model_slug(path: Path) -> str:
    name = path.resolve().stem
    cleaned = "".join(character if character.isalnum() or character in "._-" else "-" for character in name)
    return cleaned.strip("-.") or "model"


def run_directory(settings: LabellingSettings, slug: str) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{slug}-"
    existing = []
    for path in settings.output_dir.glob(f"{prefix}*"):
        suffix = path.name.removeprefix(prefix)
        if path.is_dir() and suffix.isdigit():
            existing.append(path)
    existing.sort(key=lambda path: int(path.name.removeprefix(prefix)))
    if settings.resume and existing and not (existing[-1] / COMPLETE_NAME).is_file():
        return existing[-1]
    number = int(existing[-1].name.removeprefix(prefix)) + 1 if existing else 1
    destination = settings.output_dir / f"{prefix}{number:02d}"
    destination.mkdir()
    return destination


def finished_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["ID"] for row in csv.DictReader(handle)}


def append_labels(path: Path, rows: list[dict]) -> None:
    new_file = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "path", "timestamp", "text"])
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def load_teacher(settings: LabellingSettings):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not settings.teacher_model.exists():
        raise SystemExit(f"Teacher model not found: {settings.teacher_model}")
    source = settings.teacher_model
    if source.is_file() and source.suffix == ".ckpt":
        processor = WhisperProcessor.from_pretrained(settings.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(settings.base_model)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        weights = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        if not weights:
            raise SystemExit(f"No model weights found in {source}")
        model.load_state_dict(weights)
        return model, processor
    processor = WhisperProcessor.from_pretrained(source)
    model = WhisperForConditionalGeneration.from_pretrained(source)
    return model, processor


def transcribe(model, processor, waveforms: list, settings: LabellingSettings) -> list[str]:
    import torch

    extracted = processor.feature_extractor(
        waveforms,
        sampling_rate=settings.sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    features = extracted.input_features.to(model.device)
    mask = extracted.attention_mask.to(model.device)
    use_bf16 = settings.precision.startswith("bf16")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        predicted = model.generate(
            features,
            attention_mask=mask,
            language=settings.language,
            task=settings.task,
            num_beams=1,
            do_sample=False,
        )
    return processor.tokenizer.batch_decode(predicted, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def timestamp_text(segment: Segment) -> str:
    if segment.start is None or segment.end is None:
        return ""
    return f"{segment.start:.3f}-{segment.end:.3f}"


def flush_batch(model, processor, settings: LabellingSettings, labels_path: Path, batch_audio: list, batch_rows: list[dict]) -> int:
    if not batch_audio:
        return 0
    texts = transcribe(model, processor, batch_audio, settings)
    for row, text in zip(batch_rows, texts):
        row["text"] = text
    append_labels(labels_path, batch_rows)
    count = len(batch_rows)
    batch_audio.clear()
    batch_rows.clear()
    return count


def main() -> None:
    settings = load_labelling()
    pin_gpus(settings)

    import torch
    from tqdm import tqdm

    torch.set_float32_matmul_precision("high")
    model, processor = load_teacher(settings)
    model.config.use_cache = False
    model.generation_config.language = settings.language
    model.generation_config.task = settings.task
    model.generation_config.forced_decoder_ids = None
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()

    allowed = ukrainian_paths(settings)
    segments = load_segments(settings, allowed)
    if not segments:
        raise SystemExit("No timestamp rows belong to the Ukrainian metadata.")
    by_path: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        by_path[segment.path].append(segment)

    slug = model_slug(settings.teacher_model)
    destination = run_directory(settings, slug)
    labels_path = destination / LABELS_NAME
    done = finished_ids(labels_path)
    pending_paths = {
        audio_path
        for audio_path, pieces in by_path.items()
        if any(piece.id not in done for piece in pieces)
    }
    print(
        f"segments {len(segments)} files {len(by_path)} pending {len(pending_paths)} -> {destination}",
        flush=True,
    )
    (destination / "run.json").write_text(
        json.dumps(
            {
                "teacher_model": str(settings.teacher_model),
                "base_model": settings.base_model,
                "language": settings.language,
                "sample_rate": settings.sample_rate,
                "batch_size": settings.batch_size,
                "gpus": [f"nvidia{index}" for index in settings.gpus],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shards = voa_shards(settings)
    location = index_shards(shards)
    absent = sorted(path for path in pending_paths if path not in location)
    if absent:
        raise SystemExit(f"{len(absent)} audio paths are not in the VOA shards, first: {absent[0]}")
    by_shard: dict[Path, set[str]] = defaultdict(set)
    for audio_path in pending_paths:
        by_shard[location[audio_path]].add(audio_path)

    labelled = 0
    batch_audio: list = []
    batch_rows: list[dict] = []
    files = tqdm(total=len(pending_paths), desc="files", unit="file", position=1, file=sys.stderr)
    for shard, names in tqdm(by_shard.items(), desc="shards", unit="shard", position=0, file=sys.stderr):
        for audio_path, audio_bytes in iter_shard_audio(shard, names):
            files.set_postfix_str(audio_path, refresh=False)
            waveform = decode(audio_bytes, settings.sample_rate)
            for segment in by_path[audio_path]:
                if segment.id in done:
                    continue
                audio = slice_audio(waveform, segment, settings.sample_rate, settings.max_audio_seconds)
                if audio.size == 0:
                    append_labels(
                        labels_path,
                        [{
                            "ID": segment.id,
                            "path": segment.path,
                            "timestamp": timestamp_text(segment),
                            "text": "",
                        }],
                    )
                    labelled += 1
                    continue
                batch_audio.append(audio)
                batch_rows.append(
                    {
                        "ID": segment.id,
                        "path": segment.path,
                        "timestamp": timestamp_text(segment),
                        "text": "",
                    }
                )
                if len(batch_audio) == settings.batch_size:
                    labelled += flush_batch(model, processor, settings, labels_path, batch_audio, batch_rows)
            files.update(1)
    labelled += flush_batch(model, processor, settings, labels_path, batch_audio, batch_rows)
    files.close()
    (destination / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")
    print(f"labelled {labelled} new slices -> {labels_path}", flush=True)


if __name__ == "__main__":
    main()
