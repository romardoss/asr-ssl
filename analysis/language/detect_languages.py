"""Detect the spoken language of every labelled dataset with Whisper and MMS.

The logic is the one that selects Ukrainian VOA clips (``pipeline.language``):

1. Whisper: ``scripts/monitor/meta_collector.py``. faster-whisper ``medium``,
   the first 120 seconds, VAD filter, language detected over that window.
   Only the language is needed, so the transcript is never decoded.
2. MMS: ``scripts/monitor/predict_languages.py``. ``pipeline.mms_model``
   (``facebook/mms-lid-256``), the first 300 seconds, batches packed by
   duration, fp16 autocast when ``pipeline.mms_fp16`` is on.
3. A clip is Ukrainian when both agree: Whisper says ``uk`` and the MMS argmax
   is ``ukr`` (``scripts/monitor/split_language_meta.py``).

Clips come from ``fine-tune/speech_data.py``, so they are exactly the clips
training and evaluation read. Every split of every dataset in
``fine-tune/dataset_roles.py`` that exists on disk is checked. ``--per-split N``
takes a random sample of N clips per dataset/split instead (the same sample as
``analysis/label-judge/judge_labels.py`` with the same N and ``--seed``).

Writes ``analysis/language/predictions.csv``, one row per clip, appended as
chunks finish. A stopped run continues where it stopped; clips that failed are
tried again. At the end ``language_report.py`` builds ``report.md`` and
``plots/``.

    python analysis/language/detect_languages.py
    python analysis/language/detect_languages.py --per-split 1000
    python analysis/language/detect_languages.py --datasets cv10_test fleurs --splits test
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "fine-tune", ROOT / "scripts" / "monitor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_roles import ROLES  # noqa: E402
from project_config import config_int, config_text, config_value, load_config, pin_gpu  # noqa: E402

pin_gpu()

CONFIG = load_config()
OUTPUT_PATH = HERE / "predictions.csv"
SPLITS = ("train", "dev", "test")
SAMPLING_RATE = 16000
WHISPER_SECONDS = 120  # meta_collector.LANGUAGE_SECONDS
MMS_SECONDS = 300  # predict_languages.MAX_CLIP_SECONDS
WHISPER_UKRAINIAN = "uk"  # split_language_meta.WHISPER_UKRAINIAN
MMS_UKRAINIAN = "ukr"  # split_language_meta.MMS_UKRAINIAN
TOP_LANGUAGES = 3

COLUMNS = [
    "dataset",
    "split",
    "id",
    "seconds",
    "whisper_language",
    "whisper_probability",
    "whisper_uk",
    "whisper_ru",
    "mms_language",
    "mms_probability",
    "mms_ukr",
    "mms_rus",
    "mms_top",
    "ukrainian",
    "error",
    "text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(ROLES), choices=list(ROLES))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--per-split", type=int, default=0, help="Random clips per dataset/split. 0: all (default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"Default: {OUTPUT_PATH.relative_to(ROOT)}.")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model, as in meta_collector.py.")
    parser.add_argument("--mms-model", default=config_text(CONFIG, "pipeline.mms_model"))
    parser.add_argument("--batch-size", type=int, default=config_int(CONFIG, "pipeline.mms_batch_size"))
    parser.add_argument(
        "--max-batch-seconds",
        type=float,
        default=300.0,
        help="MMS: max padded audio in one batch (longest clip times clip count).",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=bool(config_value(CONFIG, "pipeline.mms_fp16")),
        help="MMS fp16 autocast on GPU. Default: pipeline.mms_fp16.",
    )
    parser.add_argument("--chunk", type=int, default=256, help="Clips decoded and written together.")
    parser.add_argument("--no-report", action="store_true", help="Do not build report.md at the end.")
    args = parser.parse_args()
    if args.per_split < 0:
        parser.error("--per-split must be 0 or positive")
    return args


# ---------------------------------------------------------------- data


def build_work(datasets: list[str], splits: list[str], per_split: int, seed: int) -> list[tuple[str, str, object, list[int]]]:
    """(dataset, split, part, indices) for every split that exists on disk."""
    from settings import load_settings
    from speech_data import _part

    settings = load_settings()
    work = []
    for name in datasets:
        for split in splits:
            if split not in ROLES[name]:
                continue
            try:
                part = _part(settings, name, split)
            except SystemExit as error:
                print(f"skip {name}/{split}: {error}", file=sys.stderr, flush=True)
                continue
            indices = list(range(len(part)))
            if per_split and len(indices) > per_split:
                indices = sorted(random.Random(f"{seed}:{name}:{split}").sample(indices, per_split))
            print(f"{name}/{split}: {len(indices)} of {len(part)}", flush=True)
            work.append((name, split, part, indices))
    return work


def finished(path: Path) -> set[tuple[str, str, str]]:
    """Clips already written without an error."""
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("error"):
                done.add((row["dataset"], row["split"], row["id"]))
    print(f"resuming: {len(done)} clips already done in {path}", flush=True)
    return done


def append_rows(path: Path, rows: list[dict]) -> None:
    new = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def load_clip(part, index: int):
    """(text, 16 kHz mono float32 waveform) of one clip, read the way training reads it."""
    import torchaudio

    text, waveform, sample_rate = part.read(index)
    if sample_rate != SAMPLING_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLING_RATE)
    return " ".join(str(text).split()), waveform.numpy().astype("float32", copy=False)


# ---------------------------------------------------------------- models


def whisper_language(model, audio) -> dict:
    """Language of ``audio`` with the call ``meta_collector.transcribe`` makes.

    faster-whisper detects the language inside ``transcribe`` and returns the
    segments lazily. They are never iterated, so no text is decoded.
    """
    seconds_per_segment = model.feature_extractor.n_samples / model.feature_extractor.sampling_rate
    _segments, info = model.transcribe(
        audio[: WHISPER_SECONDS * SAMPLING_RATE],
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        language_detection_segments=max(1, round(WHISPER_SECONDS / seconds_per_segment)),
    )
    probabilities = dict(getattr(info, "all_language_probs", None) or [])
    probability = None if info.language_probability is None else float(info.language_probability)
    if not probabilities and info.language is not None:
        probabilities = {info.language: probability}
    return {
        "whisper_language": info.language,
        "whisper_probability": probability,
        "whisper_uk": probabilities.get("uk", 0.0 if probabilities else None),
        "whisper_ru": probabilities.get("ru", 0.0 if probabilities else None),
    }


def mms_batches(lengths: dict[int, int], max_items: int, max_seconds: float) -> list[list[int]]:
    """Pack clip keys sorted by length, as ``predict_languages.make_batches`` does."""
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0.0
    for key in sorted(lengths, key=lengths.get):
        seconds = min(lengths[key] / SAMPLING_RATE, MMS_SECONDS)
        candidate = max(longest, seconds)
        if current and (len(current) + 1 > max_items or candidate * (len(current) + 1) > max_seconds):
            batches.append(current)
            current, longest = [key], seconds
        else:
            current.append(key)
            longest = candidate
    if current:
        batches.append(current)
    return batches


class Mms:
    def __init__(self, model_id: str, fp16: bool) -> None:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.autocast = fp16 and self.device.type == "cuda"
        if int(self.processor.sampling_rate) != SAMPLING_RATE:
            raise SystemExit(f"{model_id} expects {self.processor.sampling_rate} Hz, not {SAMPLING_RATE}")
        print(f"mms {model_id} device={self.device} fp16={self.autocast}", flush=True)

    def probabilities(self, waveforms: list):
        torch = self.torch
        inputs = self.processor(
            [audio[: MMS_SECONDS * SAMPLING_RATE] for audio in waveforms],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=self.autocast):
                logits = self.model(**inputs).logits
            return torch.softmax(logits.float(), dim=-1).cpu()
        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if len(waveforms) == 1:
                raise
            middle = len(waveforms) // 2
            return torch.cat([self.probabilities(waveforms[:middle]), self.probabilities(waveforms[middle:])])

    def describe(self, probabilities) -> dict:
        values, indices = probabilities.topk(TOP_LANGUAGES)
        top = [(self.labels[int(index)], float(value)) for value, index in zip(values, indices)]
        by_label = dict(zip(self.labels, probabilities.tolist()))
        return {
            "mms_language": top[0][0],
            "mms_probability": top[0][1],
            "mms_ukr": by_label.get(MMS_UKRAINIAN),
            "mms_rus": by_label.get("rus"),
            "mms_top": " ".join(f"{label}:{value:.3f}" for label, value in top),
        }


def load_whisper(model_name: str):
    from faster_whisper import WhisperModel

    from meta_collector import use_cuda

    device = "cuda" if use_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"whisper {model_name} {device} {compute_type}", flush=True)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


# ---------------------------------------------------------------- run


def run_chunk(dataset: str, split: str, part, indices: list[int], whisper, mms: Mms, args) -> list[dict]:
    rows: dict[int, dict] = {}
    audio: dict[int, object] = {}
    for index in indices:
        row = {"dataset": dataset, "split": split, "id": part.example_id(index), "error": ""}
        try:
            text, waveform = load_clip(part, index)
            row.update({"text": text, "seconds": round(len(waveform) / SAMPLING_RATE, 3)})
            audio[index] = waveform
        except Exception as error:  # noqa: BLE001
            row["error"] = f"decode: {type(error).__name__}: {error}"
        rows[index] = row

    for index, waveform in audio.items():
        try:
            rows[index].update(whisper_language(whisper, waveform))
        except Exception as error:  # noqa: BLE001
            rows[index]["error"] = f"whisper: {type(error).__name__}: {error}"

    lengths = {index: len(waveform) for index, waveform in audio.items()}
    for batch in mms_batches(lengths, args.batch_size if mms.device.type == "cuda" else 1, args.max_batch_seconds):
        try:
            probabilities = mms.probabilities([audio[index] for index in batch])
            for index, values in zip(batch, probabilities):
                rows[index].update(mms.describe(values))
        except Exception as error:  # noqa: BLE001
            for index in batch:
                rows[index]["error"] = (rows[index]["error"] + "; " if rows[index]["error"] else "") + (
                    f"mms: {type(error).__name__}: {error}"
                )

    for row in rows.values():
        if not row["error"]:
            row["ukrainian"] = row["whisper_language"] == WHISPER_UKRAINIAN and row["mms_language"] == MMS_UKRAINIAN
    return [rows[index] for index in indices]


def main() -> None:
    args = parse_args()
    from tqdm.auto import tqdm

    work = build_work(args.datasets, args.splits, args.per_split, args.seed)
    if not work:
        raise SystemExit("No dataset split found on disk.")
    done = finished(args.output)
    whisper = load_whisper(args.whisper_model)
    mms = Mms(args.mms_model, args.fp16)

    for dataset, split, part, indices in work:
        todo = [index for index in indices if (dataset, split, part.example_id(index)) not in done]
        if not todo:
            print(f"{dataset}/{split}: done", flush=True)
            continue
        chunks = [todo[start : start + args.chunk] for start in range(0, len(todo), args.chunk)]
        errors = 0
        for chunk in tqdm(chunks, desc=f"{dataset}/{split}", unit="chunk"):
            rows = run_chunk(dataset, split, part, chunk, whisper, mms, args)
            errors += sum(1 for row in rows if row["error"])
            append_rows(args.output, rows)
        print(f"{dataset}/{split}: {len(todo)} clips, {errors} errors", flush=True)
    print(f"wrote {args.output}", flush=True)

    if not args.no_report:
        from language_report import build_report

        print(f"report: {build_report(args.output, HERE)}", flush=True)


if __name__ == "__main__":
    from meta_collector import apply_dotenv

    # faster-whisper needs the CUDA libraries from .env, which the linker only reads at startup.
    if apply_dotenv():
        os.execv(sys.executable, [sys.executable, *sys.argv])
    main()
