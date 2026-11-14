"""Fine-tune a Whisper student on Ukrainian VOA pseudo-labels.

Training rows are the slices in ``student.labels``. When ``student.filter_marks``
is true, a row is kept only if its judge mark in ``student.marks`` is one of
``student.accepted_mark``. ``latest`` in either path means the newest file for
``pseudo_labelling.teacher_model``. Audio is decoded and cut
with the same functions as pseudo-labelling. ``student.extra_train`` adds
labelled train sets (``data.train`` names, e.g. ``common_voice``) to the same
run: their clips are shuffled into batches of their own that come after all
VOA batches, so every epoch ends on them. The marks filter does not apply. Dev is scored every
``student.val_every`` fraction of an epoch, and the Common Voice test split
is scored once after training. Checkpoints go to
``paths.checkpoints/<student.name>_01/``, then ``_02``, and so on, every
``checkpoint_every`` fraction of an epoch. ``student.resume`` continues
an unfinished newest directory. A finished run starts the next directory from
``student.init``. The unnumbered ``<student.name>`` directory is the oldest
run and is left in place. The encoder stays frozen.

    python fine-tune/fine-tune-student.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
ROOT = FINE_TUNE_DIR.parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from settings import load_student_settings

SETTINGS = load_student_settings()


def _pin_gpus() -> tuple[int, ...]:
    selected = SETTINGS.gpus
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in selected)
    return selected


PHYSICAL_GPUS = _pin_gpus()

import csv
import json
import logging
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import av
import jiwer
import lightning as L
import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from checkpoints import BatchFractionCheckpoint, remove_mid_epoch, student_checkpoint_run
from common_voice import load_splits
from init_weights import load_init_model
from run_meta import chain_fields, cv_train_rows, read_meta, run_note
import tracking
from leak_filter import LEAK_FILTERED
from project_config import load_config
from settings import load_settings
from speech_data import SpeechSplit, load_train_split

sys.path.insert(0, str(ROOT / "pseudo-labelling"))
from preprocess import Segment, decode, parse_timestamp, slice_audio

MAX_AUDIO_SECONDS = SETTINGS.max_audio_seconds
MAX_LABEL_TOKENS = SETTINGS.max_label_tokens

_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class SliceExample:
    id: str
    path: str
    start: float | None
    end: float | None
    text: str


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def corpus_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    pairs = [(normalize_text(ref), normalize_text(hyp)) for ref, hyp in zip(references, hypotheses)]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref]
    if not pairs:
        return {}
    refs, hyps = zip(*pairs)
    words = jiwer.process_words(list(refs), list(hyps))
    return {
        "wer": words.wer,
        "cer": jiwer.cer(list(refs), list(hyps)),
        "mer": words.mer,
        "wil": words.wil,
        "wip": words.wip,
        "substitutions": float(words.substitutions),
        "deletions": float(words.deletions),
        "insertions": float(words.insertions),
        "hits": float(words.hits),
        "n": float(len(pairs)),
    }


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        sample_rate = int(stream.sample_rate)
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    finally:
        container.close()
    if not chunks:
        raise RuntimeError(f"No audio frames in {path}")
    audio = np.concatenate(chunks, axis=1)
    audio = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(audio)), sample_rate


def accepted_ids(path: Path, marks: tuple[int, ...]) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"Marks file not found: {path}")
    accepted = set(marks)
    latest: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latest[row["ID"]] = int(row["mark"])
    return {label_id for label_id, score in latest.items() if score in accepted}


def load_slices(labels_path: Path, wanted: set[str] | None) -> list[SliceExample]:
    if not labels_path.is_file():
        raise SystemExit(f"Labels file not found: {labels_path}")
    found: dict[str, SliceExample] = {}
    present: set[str] = set()
    empty = 0
    with labels_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_id = row["ID"]
            present.add(label_id)
            if wanted is not None and label_id not in wanted:
                continue
            if label_id in found:
                continue
            text = row["text"].strip()
            if not text:
                empty += 1
                continue
            start, end = parse_timestamp(row["timestamp"])
            found[label_id] = SliceExample(label_id, row["path"], start, end, text)
    if empty:
        print(f"skipped {empty} accepted labels with empty text", flush=True)
    if wanted is not None:
        absent = [label_id for label_id in wanted if label_id not in present]
        if absent:
            raise SystemExit(f"{len(absent)} accepted ids are missing from {labels_path}, first: {absent[0]}")
    return list(found.values())


def voa_shards(root: Path) -> list[Path]:
    found = sorted((root / "data").glob("train-*.parquet"))
    if not found:
        raise SystemExit(f"No train parquet shards in {root / 'data'}")
    return found


def locate_slices(shards: list[Path], needed: set[str]) -> dict[str, tuple[Path, int]]:
    def one(shard: Path) -> list[tuple[str, int]]:
        parquet = pq.ParquetFile(shard)
        names = pq.read_table(shard, columns=["audio.path"]).column("path").to_pylist()
        hits: list[tuple[str, int]] = []
        offset = 0
        for row_group in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(row_group).num_rows
            for name in names[offset : offset + count]:
                if name in needed:
                    hits.append((str(name), row_group))
            offset += count
        return hits

    located: dict[str, tuple[Path, int]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shard, hits in tqdm(
            zip(shards, pool.map(one, shards)),
            total=len(shards),
            desc="index shards",
            file=sys.stderr,
        ):
            for name, row_group in hits:
                located[name] = (shard, row_group)
    missing = sorted(needed.difference(located))
    if missing:
        raise SystemExit(f"{len(missing)} training audios are not in the VOA shards, first: {missing[0]}")
    return located


def read_row_group_audio(shard: Path, row_group: int) -> dict[str, bytes]:
    table = pq.ParquetFile(shard).read_row_group(row_group, columns=["audio"])
    return {str(item["path"]): item["bytes"] for item in table.column("audio").to_pylist()}


class RowGroupSampler(Sampler[int]):
    """Yield slice indexes with each VOA file's slices kept together.

    Row groups stay in one stretch, so a worker can read a parquet group once
    and decode each file once. Each GPU receives an equal contiguous share.

    ``extra`` labelled clips (indexes ``extra_offset`` and up) are shuffled
    into batches of their own that come after every VOA batch, so each epoch
    ends on the labelled data. Each GPU gets an equal share of both parts.
    """

    def __init__(
        self,
        groups: list[list[list[int]]],
        batch_size: int,
        seed: int,
        extra: int = 0,
        extra_offset: int = 0,
    ):
        self.groups = groups
        self.batch_size = batch_size
        self.seed = seed
        self.extra = extra
        self.extra_offset = extra_offset
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.groups)))
        rng.shuffle(order)
        flat: list[int] = []
        for group_index in order:
            files = [list(indexes) for indexes in self.groups[group_index]]
            rng.shuffle(files)
            flat.extend(index for indexes in files for index in indexes)
        labelled = list(range(self.extra_offset, self.extra_offset + self.extra))
        rng.shuffle(labelled)
        world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        rank = int(os.environ.get("RANK", "0"))
        voa = self._share(flat, world, rank)
        if not voa:
            raise RuntimeError("Not enough student slices to fill one batch on every GPU")
        # The labelled batches go last in the epoch.
        return voa + self._share(labelled, world, rank)

    def _share(self, indexes: list[int], world: int, rank: int) -> list[int]:
        """This rank's contiguous share of whole batches."""
        size = self.batch_size
        per_rank = (len(indexes) // (world * size)) * size
        return indexes[rank * per_rank : (rank + 1) * per_rank]

    def __iter__(self):
        return iter(self._indices())

    def __len__(self) -> int:
        return len(self._indices())


class VoaSliceDataset(Dataset):
    def __init__(
        self,
        examples: list[SliceExample],
        locations: dict[str, tuple[Path, int]],
        processor: WhisperProcessor,
        lang_token_id: int,
        sample_rate: int,
    ):
        self.examples = examples
        self.locations = locations
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.sample_rate = sample_rate
        transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.prefix_ids = [lang_token_id, transcribe_id, notimestamps_id]
        self.eos_token_id = self.tokenizer.eos_token_id
        self._group_key: tuple[str, int] | None = None
        self._group_audio: dict[str, bytes] = {}
        self._wave_path: str | None = None
        self._wave: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def _waveform(self, example: SliceExample) -> np.ndarray:
        if example.path == self._wave_path and self._wave is not None:
            return self._wave
        shard, row_group = self.locations[example.path]
        key = (str(shard), row_group)
        if key != self._group_key:
            self._group_audio = read_row_group_audio(shard, row_group)
            self._group_key = key
            self._wave_path = None
        self._wave = decode(self._group_audio[example.path], self.sample_rate)
        self._wave_path = example.path
        return self._wave

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        audio = slice_audio(
            self._waveform(example),
            Segment(example.id, example.path, example.start, example.end),
            self.sample_rate,
            MAX_AUDIO_SECONDS,
        )
        if audio.size == 0:
            raise RuntimeError(f"Empty audio for {example.id}")
        extracted = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = self.tokenizer(example.text, add_special_tokens=False).input_ids
        text_ids = text_ids[: MAX_LABEL_TOKENS - len(self.prefix_ids) - 1]
        labels = self.prefix_ids + text_ids + [self.eos_token_id]
        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
        }


class CommonVoiceDataset(Dataset):
    def __init__(self, frame, processor: WhisperProcessor, lang_token_id: int):
        self.frame = frame.reset_index(drop=True)
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.sampling_rate = processor.feature_extractor.sampling_rate
        transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.prefix_ids = [lang_token_id, transcribe_id, notimestamps_id]
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        waveform, sample_rate = load_waveform(row["audio_path"])
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_rate)
        max_samples = int(MAX_AUDIO_SECONDS * self.sampling_rate)
        if waveform.numel() > max_samples:
            waveform = waveform[:max_samples]
        extracted = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = self.tokenizer(str(row["sentence"]), add_special_tokens=False).input_ids
        text_ids = text_ids[: MAX_LABEL_TOKENS - len(self.prefix_ids) - 1]
        labels = self.prefix_ids + text_ids + [self.eos_token_id]
        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
        }


class LabelledDataset(Dataset):
    """Clips of the ``student.extra_train`` sets, prepared like the VOA slices."""

    def __init__(self, split: SpeechSplit, processor: WhisperProcessor, lang_token_id: int):
        self.split = split
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.sampling_rate = processor.feature_extractor.sampling_rate
        transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.prefix_ids = [lang_token_id, transcribe_id, notimestamps_id]
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict:
        sentence, waveform, sample_rate = self.split.read(index)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_rate)
        max_samples = int(MAX_AUDIO_SECONDS * self.sampling_rate)
        if waveform.numel() > max_samples:
            waveform = waveform[:max_samples]
        extracted = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = self.tokenizer(sentence, add_special_tokens=False).input_ids
        text_ids = text_ids[: MAX_LABEL_TOKENS - len(self.prefix_ids) - 1]
        labels = self.prefix_ids + text_ids + [self.eos_token_id]
        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
        }


def collate_fn(features: list[dict]) -> dict:
    max_label_len = max(len(feature["labels"]) for feature in features)
    labels = torch.full((len(features), max_label_len), -100, dtype=torch.long)
    for index, feature in enumerate(features):
        labels[index, : len(feature["labels"])] = torch.tensor(feature["labels"], dtype=torch.long)
    return {
        "input_features": torch.stack([feature["input_features"] for feature in features]),
        "attention_mask": torch.stack([feature["attention_mask"] for feature in features]),
        "labels": labels,
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WhisperStudent(L.LightningModule):
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        lang_token_id: int,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = processor.tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model", "processor"])

        for parameter in self.model.model.encoder.parameters():
            parameter.requires_grad = False
        self.model.train()
        self.model.model.encoder.eval()
        self.language = processor.tokenizer.convert_ids_to_tokens(lang_token_id).strip("<|>")
        self.model.generation_config.language = self.language
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.val_references: list[str] = []
        self.val_hypotheses: list[str] = []
        self.test_references: list[str] = []
        self.test_hypotheses: list[str] = []
        self.val_metrics: dict[str, float] = {}
        self.test_metrics: dict[str, float] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.model.encoder.eval()
        return self

    def training_step(self, batch, batch_idx):
        loss = self.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self) -> None:
        self.train(True)
        if self.trainer.is_global_zero:
            logging.getLogger("fine-tune-student").info(
                "epoch %d decoder_training=%s encoder_training=%s",
                self.current_epoch,
                self.model.model.decoder.training,
                self.model.model.encoder.training,
            )

    def _transcribe(self, batch) -> tuple[list[str], list[str]]:
        predicted_ids = self.model.generate(
            batch["input_features"],
            attention_mask=batch["attention_mask"],
            language=self.language,
            task="transcribe",
            num_beams=1,
            do_sample=False,
            use_cache=True,
        )
        hypotheses = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        references = self.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return references, hypotheses

    def validation_step(self, batch, batch_idx):
        loss = self.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        references, hypotheses = self._transcribe(batch)
        self.val_references.extend(references)
        self.val_hypotheses.extend(hypotheses)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        references, hypotheses = self._transcribe(batch)
        self.test_references.extend(references)
        self.test_hypotheses.extend(hypotheses)
        return loss

    def _gather_texts(self, references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
        if self.trainer.world_size == 1:
            return references, hypotheses
        gathered_refs: list[list[str] | None] = [None] * self.trainer.world_size
        gathered_hyps: list[list[str] | None] = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(gathered_refs, references)
        torch.distributed.all_gather_object(gathered_hyps, hypotheses)
        return (
            [text for part in gathered_refs if part for text in part],
            [text for part in gathered_hyps if part for text in part],
        )

    def _log_corpus(self, prefix: str, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        references, hypotheses = self._gather_texts(references, hypotheses)
        metrics = corpus_metrics(references, hypotheses)
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, prog_bar=name in {"wer", "cer"}, sync_dist=True)
        return metrics

    def on_validation_epoch_end(self):
        self.val_metrics = self._log_corpus("val", self.val_references, self.val_hypotheses)
        self.val_references.clear()
        self.val_hypotheses.clear()

    def on_test_epoch_end(self):
        self.test_metrics = self._log_corpus("test", self.test_references, self.test_hypotheses)
        self.test_references.clear()
        self.test_hypotheses.clear()

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_examples(examples: list[SliceExample], locations: dict[str, tuple[Path, int]]) -> list[list[list[int]]]:
    grouped: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, example in enumerate(examples):
        shard, row_group = locations[example.path]
        grouped[(str(shard), row_group)][example.path].append(index)
    return [list(files.values()) for files in grouped.values()]


def main() -> None:
    cfg = SETTINGS
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_dir = cfg.output_dir / cfg.name
    if local_rank == 0:
        setup_logging(output_dir / "train.log")
    log = logging.getLogger("fine-tune-student")

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    device_ids = list(range(len(PHYSICAL_GPUS)))
    device_names = [f"nvidia{index}" for index in PHYSICAL_GPUS]
    log.info(
        "gpus %s CUDA_VISIBLE_DEVICES=%s train_batch=%d eval_batch=%d",
        device_names,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        cfg.train_batch_size,
        cfg.eval_batch_size,
    )

    if cfg.filter_marks:
        wanted = accepted_ids(cfg.marks, cfg.accepted_mark)
        examples = load_slices(cfg.labels, wanted)
        if not examples:
            raise SystemExit(f"No labels with marks {', '.join(str(mark) for mark in cfg.accepted_mark)}")
    else:
        examples = load_slices(cfg.labels, None)
        if not examples:
            raise SystemExit(f"No usable labels in {cfg.labels}")
    locations = locate_slices(voa_shards(cfg.voa), {example.path for example in examples})
    log.info("student slices: %d files: %d", len(examples), len({example.path for example in examples}))

    splits = load_splits()
    log.info("common voice dev clips: %d", len(splits["dev"]))
    log.info("common voice test clips: %d", len(splits["test"]))

    processor = WhisperProcessor.from_pretrained(cfg.base)
    if processor.feature_extractor.sampling_rate != cfg.sample_rate:
        raise SystemExit(
            f"student.sample_rate is {cfg.sample_rate}, Whisper expects {processor.feature_extractor.sampling_rate}"
        )
    lang_token = f"<|{cfg.language}|>"
    lang_token_id = processor.tokenizer.convert_tokens_to_ids(lang_token)
    unk_id = processor.tokenizer.unk_token_id
    if lang_token_id is None or lang_token_id == unk_id:
        raise SystemExit(f"{cfg.base} has no language token {lang_token}")

    model = load_init_model(cfg.init, cfg.base)
    model.config.use_cache = False

    train_dataset = VoaSliceDataset(examples, locations, processor, lang_token_id, cfg.sample_rate)
    extra_clips = 0
    if cfg.extra_train:
        # Paths of the labelled sets come from the same config file.
        labelled = LabelledDataset(load_train_split(cfg.extra_train), processor, lang_token_id)
        extra_clips = len(labelled)
        for part in labelled.split.parts:
            log.info("extra train %s: %d clips", part.source, len(part))
        train_dataset = ConcatDataset([train_dataset, labelled])
    train_sampler = RowGroupSampler(
        group_examples(examples, locations),
        cfg.train_batch_size,
        cfg.seed,
        extra=extra_clips,
        extra_offset=len(examples),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", "0"))

    def eval_loader(frame) -> DataLoader:
        dataset = CommonVoiceDataset(frame, processor, lang_token_id)
        sampler = None
        if world > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world, rank=rank, shuffle=False
            )
        return DataLoader(
            dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.num_workers,
            worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=cfg.num_workers > 0,
        )

    dev_loader = eval_loader(splits["dev"])
    test_loader = eval_loader(splits["test"])

    module = WhisperStudent(
        model,
        processor=processor,
        lang_token_id=lang_token_id,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
    )
    frozen = sum(parameter.numel() for parameter in module.model.parameters() if not parameter.requires_grad)
    trainable = sum(parameter.numel() for parameter in module.model.parameters() if parameter.requires_grad)
    log.info("frozen encoder parameters: %d", frozen)
    log.info("trainable decoder parameters: %d", trainable)

    ckpt_dir, ckpt_path = student_checkpoint_run(cfg.checkpoints, cfg.name, cfg.resume, cfg.max_epochs)
    log.info("checkpoint directory %s", ckpt_dir)
    train_meta = {
        "model": cfg.name,
        "base": cfg.base,
        "init": cfg.init,
        "language": cfg.language,
        "learning_rate": cfg.learning_rate,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "precision": cfg.precision,
        "frozen": "encoder",
        "trainable": "decoder",
        "filter_marks": cfg.filter_marks,
        "accepted_mark": list(cfg.accepted_mark),
        "train_data": ["voa_pseudo_labels", *cfg.extra_train],
        # Without common_voice in extra_train the student sees Common Voice only through its init model.
        "cv_train_rows": (
            cv_train_rows(load_settings().common_voice_splits)
            if "common_voice" in cfg.extra_train
            else read_meta(Path(cfg.init).parent).get("cv_train_rows")
        ),
        "labels": str(cfg.labels),
        "stage": "student",
        **chain_fields(),
        "note": run_note(ckpt_dir),
        "resumed_from": ckpt_path.name if ckpt_path else None,
    }
    tracker = tracking.start_training(
        ckpt_dir.name,
        ckpt_dir,
        {
            **train_meta,
            **tracking.config_params(load_config(), "student"),
            "student_slices": len(examples),
            "extra_train_clips": extra_clips,
            "leak_filter": [name for name in cfg.extra_train if name in LEAK_FILTERED],
            **tracking.clip_params(splits),
        },
        {key: train_meta.get(key) for key in ("model", "stage", "init", "chain", "step", "note", "resumed_from")},
        resumed=ckpt_path is not None,
    )
    if tracker is not None:
        train_meta["mlflow_run_id"] = tracker.run_id
    checkpoint_callback = BatchFractionCheckpoint(
        ckpt_dir, cfg.name, fraction=cfg.checkpoint_every, meta=train_meta
    )
    logger = CSVLogger(save_dir=str(cfg.output_dir), name=cfg.name)
    checkpoint_callback.metrics_dir = Path(logger.log_dir)
    strategy = DDPStrategy(find_unused_parameters=False) if len(device_ids) > 1 else "auto"
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=device_ids,
        strategy=strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip,
        val_check_interval=cfg.val_every,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step"), *([tracker.callback()] if tracker else [])],
        logger=[logger, tracker.logger()] if tracker else logger,
        default_root_dir=str(output_dir),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        enable_progress_bar=True,
        use_distributed_sampler=False,
    )

    baseline_metrics: dict[str, float] = {}
    if ckpt_path is not None:
        log.info("resuming from %s", ckpt_path)
    elif cfg.baseline_eval:
        log.info("scoring the dev split before student training")
        trainer.validate(module, dataloaders=dev_loader)
        baseline_metrics = dict(module.val_metrics)

    trainer.fit(module, train_loader, dev_loader, ckpt_path=ckpt_path)
    # A finished run needs only the end of each epoch; an interrupted one keeps everything to resume.
    if cfg.remove_mid_epoch_checkpoints and trainer.is_global_zero and not trainer.interrupted:
        cleaned = remove_mid_epoch(ckpt_dir)
        if cleaned:
            log.info(cleaned)
    trainer.test(module, dataloaders=test_loader)

    if trainer.is_global_zero:
        payload = {
            "model_name": cfg.name,
            "base": cfg.base,
            "init": cfg.init,
            "language": cfg.language,
            "filter_marks": cfg.filter_marks,
            "accepted_mark": list(cfg.accepted_mark),
            "devices": device_names or device_ids,
            "epochs": cfg.max_epochs,
            "frozen": "encoder",
            "trainable": "decoder",
            "student_slices": len(examples),
            "common_voice": {"dev": len(splits["dev"]), "test": len(splits["test"])},
            "checkpoints": str(ckpt_dir),
            "resumed_from": str(ckpt_path) if ckpt_path else None,
            "dev_before_training": baseline_metrics,
            "dev": module.val_metrics,
            "test": module.test_metrics,
        }
        destination = output_dir / "results.json"
        write_results(destination, payload)
        log.info("wrote %s", destination)
        log.info("dev %s", module.val_metrics)
        log.info("test %s", module.test_metrics)
        if tracker is not None:
            logger.save()
            tracker.log_files(
                [destination, output_dir / "train.log", Path(logger.log_dir) / "metrics.csv", ckpt_dir / "meta.yaml", ckpt_dir / "config.yaml"]
            )


if __name__ == "__main__":
    main()
