"""Label Ukrainian VOA slices with the teacher model from ``config.yaml``.

Clips that are not listed in ``pseudo_labelling.ukrainian_meta`` are skipped.
Each remaining clip is resampled and cut on ``pseudo_labelling.timestamps``.
Slices already in ``pseudo_labelling.voa_cache`` (``python main.py cache-voa``)
are read from it first; then only the mp3 files that still have uncached slices
are decoded with ffmpeg.
Transcripts are appended to ``labels.csv`` under
``pseudo_labelling.output_dir/<teacher>-XX``. With ``resume: true``, the newest
unfinished folder for that teacher is continued, unless its ``labels.csv`` has
an older header: then a new folder is started.

Besides the text, every row stores the teacher's own confidence, taken from the
greedy decoding itself (no second pass, no stored vocabulary scores):

1. ``avg_logprob``: mean log-probability of the generated tokens, end of text included.
2. ``confidence``: ``exp(avg_logprob)``, the geometric mean token probability (0-1).
3. ``min_logprob``: log-probability of the least likely token.
4. ``num_tokens``: generated text tokens, end of text excluded.
5. ``compression_ratio``: UTF-8 bytes / zlib-compressed bytes of the text; repeated loops score high.
6. ``truncated``: 1 when the token budget of the slice ended decoding (see ``token_budget``).

    python pseudo-labelling/label.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import zlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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

# Appended, not prepended: fine-tune/settings.py must not shadow pseudo-labelling/settings.py.
_FINE_TUNE = str(Path(__file__).resolve().parents[1] / "fine-tune")
if _FINE_TUNE not in sys.path:
    sys.path.append(_FINE_TUNE)
from voa_slices import cache_file, read_cached, usable_cache  # noqa: E402

LABELS_NAME = "labels.csv"
COMPLETE_NAME = "complete"
# ffmpeg runs in its own process, so threads decode in parallel while the GPU generates.
DECODE_WORKERS = min(8, os.cpu_count() or 1)
# Files decoded ahead of the one being labelled; bounds the waveforms held in memory.
DECODE_AHEAD = 64
# Cached slices read ahead of the one being labelled (one 30 s slice is about 2 MB as float32).
CACHE_AHEAD = 256
# Threads that check which slices are cached; a stat per slice is slow on network disks.
STAT_WORKERS = 32
# Slices are sorted by duration within this many batches, so a batch holds slices
# of similar length and short ones do not wait for the longest transcript.
SORT_WINDOW_BATCHES = 4
# Token budget per slice: ceil(seconds * rate) + base, at most MAX_NEW_TOKENS.
# On the 351 904 VOA slices labelled by whisper-uk-epoch00, real speech stays under
# 9.2 tokens/s at the 99th percentile, while 2 909 slices (0.8 %) looped up to the
# 448-token limit and kept 38 % of 64-slice batches decoding to the end. At 12 tokens/s
# + 16 the budget cuts all those loops and 163 other slices, which are also
# repetitions or noise ("Ааааа…", "Ми? — Ми? — …"); mean decoder steps per batch
# fall from 308 to about 225.
MAX_TOKENS_PER_SECOND = 12
TOKEN_BUDGET_BASE = 16
# Whisper's decoder holds 448 positions; the language and task prompt takes 4.
MAX_NEW_TOKENS = 440
LABEL_FIELDS = [
    "ID",
    "path",
    "timestamp",
    "text",
    "avg_logprob",
    "confidence",
    "min_logprob",
    "num_tokens",
    "compression_ratio",
    "truncated",
]


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
    if (
        settings.resume
        and existing
        and not (existing[-1] / COMPLETE_NAME).is_file()
        and header_matches(existing[-1] / LABELS_NAME)
    ):
        return existing[-1]
    number = int(existing[-1].name.removeprefix(prefix)) + 1 if existing else 1
    destination = settings.output_dir / f"{prefix}{number:02d}"
    destination.mkdir()
    return destination


def header_matches(path: Path) -> bool:
    """True when ``path`` is missing or empty, or its header is ``LABEL_FIELDS``."""
    if not path.is_file() or path.stat().st_size == 0:
        return True
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), []) == LABEL_FIELDS


def finished_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["ID"] for row in csv.DictReader(handle)}


def append_labels(path: Path, rows: list[dict]) -> None:
    new_file = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
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


def token_budget(samples: int, sample_rate: int) -> int:
    seconds = samples / sample_rate
    return min(MAX_NEW_TOKENS, math.ceil(seconds * MAX_TOKENS_PER_SECOND) + TOKEN_BUDGET_BASE)


class GreedyConfidence:
    """Logits processor that scores each slice's greedy tokens and enforces its token budget.

    Whisper's ``generate`` puts custom processors after its own, so ``scores`` here
    are final: greedy search takes their argmax, which is the token scored here.
    Per step only a few numbers per row are kept, not the vocabulary scores that
    ``output_scores=True`` would hold. A row that has spent its budget gets end of
    text forced and is marked truncated. Rows that already ended are ignored:
    ``generate`` keeps feeding them padding until the whole batch ends.
    """

    def __init__(self, budgets, eos_token_id: int):
        import torch

        self.budgets = budgets
        self.eos_token_id = eos_token_id
        self.ended = torch.zeros_like(budgets, dtype=torch.bool)
        self.truncated = torch.zeros_like(budgets, dtype=torch.bool)
        self.count = torch.zeros_like(budgets)
        self.logprob_sum = torch.zeros(budgets.shape, dtype=torch.float32, device=budgets.device)
        self.logprob_min = torch.full(budgets.shape, math.inf, dtype=torch.float32, device=budgets.device)

    def __call__(self, input_ids, scores):
        import torch

        live = ~self.ended
        force = live & self.count.ge(self.budgets)
        if bool(force.any()):
            forced = torch.full_like(scores, -math.inf)
            forced[:, self.eos_token_id] = 0.0
            scores = torch.where(force.unsqueeze(1), forced, scores)
            self.truncated |= force
        logprobs = torch.log_softmax(scores.float(), dim=-1)
        best, token = logprobs.max(dim=-1)
        scored = live & ~force
        self.logprob_sum += torch.where(scored, best, 0.0)
        self.logprob_min = torch.where(scored, torch.minimum(self.logprob_min, best), self.logprob_min)
        self.count += scored.long()
        self.ended |= live & token.eq(self.eos_token_id)
        return scores

    def rows(self) -> list[dict]:
        """Per-row confidence fields; call after ``generate`` returns."""
        count = self.count.cpu().tolist()
        total = self.logprob_sum.cpu().tolist()
        lowest = self.logprob_min.cpu().tolist()
        truncated = self.truncated.cpu().tolist()
        result = []
        for scored, logprob_sum, logprob_min, cut in zip(count, total, lowest, truncated):
            if scored == 0:
                result.append({"avg_logprob": "", "confidence": "", "min_logprob": "", "num_tokens": 0, "truncated": int(cut)})
                continue
            average = logprob_sum / scored
            result.append(
                {
                    "avg_logprob": f"{average:.5f}",
                    "confidence": f"{math.exp(average):.5f}",
                    "min_logprob": f"{logprob_min:.5f}",
                    # A row that ended on its own scored its end-of-text token too.
                    "num_tokens": scored if cut else scored - 1,
                    "truncated": int(cut),
                }
            )
        return result


def compression_ratio(text: str) -> str:
    data = text.encode("utf-8")
    if not data:
        return ""
    return f"{len(data) / len(zlib.compress(data)):.3f}"


def transcribe(model, processor, waveforms: list, settings: LabellingSettings) -> list[dict]:
    """Greedy transcripts of ``waveforms`` with ``text`` and the confidence fields of ``LABEL_FIELDS``."""
    import torch
    from transformers import LogitsProcessorList

    extracted = processor.feature_extractor(
        waveforms,
        sampling_rate=settings.sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
        # log-mel on the GPU (torch STFT); in numpy it runs on one CPU core while the GPU waits.
        device=str(model.device),
    )
    features = extracted.input_features.to(model.device, dtype=model.dtype)
    mask = extracted.attention_mask.to(model.device)
    budgets = torch.tensor(
        [token_budget(len(waveform), settings.sample_rate) for waveform in waveforms],
        dtype=torch.long,
        device=model.device,
    )
    scorer = GreedyConfidence(budgets, model.generation_config.eos_token_id)
    use_bf16 = settings.precision.startswith("bf16")
    with torch.inference_mode(), torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        predicted = model.generate(
            features,
            attention_mask=mask,
            language=settings.language,
            task=settings.task,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            logits_processor=LogitsProcessorList([scorer]),
            # The longest budget plus the forced end of text; no batch decodes past it.
            max_new_tokens=int(budgets.max()) + 1,
        )
    texts = processor.tokenizer.batch_decode(predicted, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rows = scorer.rows()
    for row, text in zip(rows, texts):
        row["text"] = text
        row["compression_ratio"] = compression_ratio(text)
    return rows


def timestamp_text(segment: Segment) -> str:
    if segment.start is None or segment.end is None:
        return ""
    return f"{segment.start:.3f}-{segment.end:.3f}"


def decoded_files(by_shard: dict[Path, set[str]], sample_rate: int, shards_bar):
    """Yield ``(audio_path, waveform)`` in shard order, decoding the next files in a thread pool."""
    pending: deque = deque()
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        for shard, names in by_shard.items():
            for audio_path, audio_bytes in iter_shard_audio(shard, names):
                pending.append((audio_path, pool.submit(decode, audio_bytes, sample_rate)))
                if len(pending) > DECODE_AHEAD:
                    audio_path, future = pending.popleft()
                    yield audio_path, future.result()
            shards_bar.update(1)
        while pending:
            audio_path, future = pending.popleft()
            yield audio_path, future.result()


def labelling_cache(settings: LabellingSettings) -> Path | None:
    """``settings.voa_cache`` if its slices match the labelling audio, else None (with the reason printed)."""
    if settings.voa_cache is None:
        print("voa_cache is not set: every slice is decoded with ffmpeg", flush=True)
        return None
    cache, reason = usable_cache(settings.voa_cache, settings.sample_rate, settings.max_audio_seconds)
    if cache is None:
        print(f"voa_cache not used: {reason}", flush=True)
    return cache


def split_cached(segments: list[Segment], cache: Path | None) -> tuple[list[Segment], list[Segment]]:
    """``(cached, uncached)`` slices, in the order of ``segments``."""
    if cache is None:
        return [], list(segments)
    with ThreadPoolExecutor(max_workers=STAT_WORKERS) as pool:
        present = list(pool.map(lambda segment: cache_file(cache, segment.id).is_file(), segments, chunksize=256))
    cached = [segment for segment, hit in zip(segments, present) if hit]
    uncached = [segment for segment, hit in zip(segments, present) if not hit]
    return cached, uncached


def _read_slice(path: Path):
    try:
        return read_cached(path)
    except (OSError, ValueError, EOFError):
        return None


def cached_slices(segments: list[Segment], cache: Path | None):
    """Yield ``(segment, waveform)`` from the cache, reading ahead in threads; waveform is None if unreadable.

    The cache holds the same ffmpeg decode, cut, and 30 s cap as ``slice_audio``,
    stored as int16, so a cached slice equals a freshly decoded one up to int16 rounding.
    """
    if not segments:
        return
    pending: deque = deque()
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        for segment in segments:
            pending.append((segment, pool.submit(_read_slice, cache_file(cache, segment.id))))
            if len(pending) > CACHE_AHEAD:
                done_segment, future = pending.popleft()
                yield done_segment, future.result()
        while pending:
            done_segment, future = pending.popleft()
            yield done_segment, future.result()


def blank_row(segment: Segment) -> dict:
    row = {field: "" for field in LABEL_FIELDS}
    row.update({"ID": segment.id, "path": segment.path, "timestamp": timestamp_text(segment)})
    return row


def flush_sorted(model, processor, settings: LabellingSettings, labels_path: Path, pending: list, keep_partial: bool) -> int:
    """Label ``pending`` ``(audio, row)`` pairs in batches of similar duration.

    With ``keep_partial`` the slices left over after the last full batch stay in
    ``pending`` to be sorted with the next ones; otherwise everything is labelled.
    """
    pending.sort(key=lambda item: len(item[0]))
    size = settings.batch_size
    stop = len(pending) - len(pending) % size if keep_partial else len(pending)
    labelled = 0
    for start in range(0, stop, size):
        chunk = pending[start : start + size]
        labelled += flush_batch(model, processor, settings, labels_path, [audio for audio, _ in chunk], [row for _, row in chunk])
    del pending[:stop]
    return labelled


def flush_batch(model, processor, settings: LabellingSettings, labels_path: Path, batch_audio: list, batch_rows: list[dict]) -> int:
    if not batch_audio:
        return 0
    for row, result in zip(batch_rows, transcribe(model, processor, batch_audio, settings)):
        row.update(result)
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
    model.generation_config.language = settings.language
    model.generation_config.task = settings.task
    model.generation_config.forced_decoder_ids = None
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    # bf16 weights: half the memory and faster than fp32 weights under autocast; no gradients here.
    model.to(device, dtype=torch.bfloat16 if settings.precision.startswith("bf16") else torch.float32)
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
    print(f"segments {len(segments)} files {len(by_path)} done {len(done)} -> {destination}", flush=True)
    (destination / "run.json").write_text(
        json.dumps(
            {
                "teacher_model": str(settings.teacher_model),
                "base_model": settings.base_model,
                "language": settings.language,
                "sample_rate": settings.sample_rate,
                "batch_size": settings.batch_size,
                "decoding": "greedy",
                "sort_window_batches": SORT_WINDOW_BATCHES,
                "token_budget": f"min({MAX_NEW_TOKENS}, ceil(seconds * {MAX_TOKENS_PER_SECOND}) + {TOKEN_BUDGET_BASE})",
                "fields": LABEL_FIELDS,
                "voa_cache": str(settings.voa_cache) if settings.voa_cache else None,
                "gpus": [f"nvidia{index}" for index in settings.gpus],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    todo = [segment for segment in segments if segment.id not in done]
    cache = labelling_cache(settings)
    cached, uncached = split_cached(todo, cache)
    print(f"slices to label {len(todo)}: cached {len(cached)}, decode with ffmpeg {len(uncached)}", flush=True)

    labelled = 0
    pending: list = []
    window = settings.batch_size * SORT_WINDOW_BATCHES
    slices_bar = tqdm(total=len(todo), desc="slices", unit="slice", file=sys.stderr)

    def add(segment: Segment, audio) -> None:
        nonlocal labelled
        if audio.size == 0:
            append_labels(labels_path, [blank_row(segment)])
            labelled += 1
        else:
            pending.append((audio, blank_row(segment)))
        slices_bar.update(1)
        if len(pending) >= window:
            labelled += flush_sorted(model, processor, settings, labels_path, pending, keep_partial=True)

    for segment, audio in cached_slices(cached, cache):
        if audio is None:
            # Unreadable cache file: decode this slice from its mp3 below.
            uncached.append(segment)
            continue
        add(segment, audio)

    if uncached:
        wanted: dict[str, list[Segment]] = defaultdict(list)
        for segment in uncached:
            wanted[segment.path].append(segment)
        location = index_shards(voa_shards(settings), settings.output_dir / "voa_shard_index.json")
        absent = sorted(path for path in wanted if path not in location)
        if absent:
            raise SystemExit(f"{len(absent)} audio paths are not in the VOA shards, first: {absent[0]}")
        by_shard: dict[Path, set[str]] = defaultdict(set)
        for audio_path in wanted:
            by_shard[location[audio_path]].add(audio_path)
        shards_bar = tqdm(total=len(by_shard), desc="shards", unit="shard", file=sys.stderr)
        for audio_path, waveform in decoded_files(by_shard, settings.sample_rate, shards_bar):
            for segment in wanted[audio_path]:
                add(segment, slice_audio(waveform, segment, settings.sample_rate, settings.max_audio_seconds))
        shards_bar.close()
    labelled += flush_sorted(model, processor, settings, labels_path, pending, keep_partial=False)
    slices_bar.close()
    (destination / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")
    print(f"labelled {labelled} new slices -> {labels_path}", flush=True)


if __name__ == "__main__":
    main()
