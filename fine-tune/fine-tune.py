"""Fine-tune a Whisper decoder on Ukrainian Common Voice.

The encoder stays frozen. Train/dev/test rows come from ``meta/common-voice``
and audio from ``$PATH_TO_COMMON_VOICE/clips``. The starting checkpoint is
``$WHISPER_MODEL``. Training uses both RTX 5090s
(``nvidia0`` = cuda:0, ``nvidia1`` = cuda:1).

Checkpoints (model, optimizer, and scheduler) are written to
``$PATH_TO_CHECKPOINTS/<model-name>/`` at the end of every epoch and after
every 5% of the training batches. If ``last.ckpt`` is already there, training
resumes from it. Metrics and logs go to ``fine-tune/output/<model-name>/``.

    python fine-tune/fine-tune.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import av
import jiwer
import lightning as L
import numpy as np
import pandas as pd
import torch
import torchaudio
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from checkpoints import BatchFractionCheckpoint, resume_checkpoint
from common_voice import checkpoint_root, load_dotenv, load_splits, whisper_model

DEFAULT_MODEL_NAME = "model-whisper-uk"
NVIDIA_DEVICES = {"nvidia0": 0, "nvidia1": 1}
MAX_AUDIO_SECONDS = 30.0
MAX_LABEL_TOKENS = 448
OUTPUT_DIR = FINE_TUNE_DIR / "output"

_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("WHISPER_MODEL", "").strip() or None,
        help="Hugging Face model id. Defaults to WHISPER_MODEL from .env.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Used in checkpoint filenames and as the directory under PATH_TO_CHECKPOINTS.",
    )
    parser.add_argument("--language", default="uk", help="Whisper language code, e.g. uk.")
    parser.add_argument(
        "--devices",
        default="nvidia0,nvidia1",
        help="Comma-separated device names (nvidia0, nvidia1) or CUDA indices.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the newest checkpoint in the model directory when one exists.",
    )
    parser.add_argument("--limit-train-batches", type=_parse_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=_parse_limit, default=1.0)
    parser.add_argument("--limit-test-batches", type=_parse_limit, default=1.0)
    parser.add_argument(
        "--baseline-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score the dev split before any updates, unless training is resumed.",
    )
    return parser.parse_args()


def _parse_limit(value: str) -> int | float:
    """Lightning treats an int as a batch count and a float as a fraction."""
    if any(token in value for token in (".", "e", "E")):
        return float(value)
    return int(value)


def resolve_devices(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        name = part.strip()
        if not name:
            continue
        if name in NVIDIA_DEVICES:
            indices.append(NVIDIA_DEVICES[name])
        elif name.isdigit():
            indices.append(int(name))
        else:
            known = ", ".join(NVIDIA_DEVICES)
            raise SystemExit(f"Unknown device {name!r}. Use {known}, or a CUDA index.")
    if not indices:
        raise SystemExit("No devices selected.")
    if len(set(indices)) != len(indices):
        raise SystemExit(f"Duplicate devices in {spec!r}.")
    return indices


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


class WhisperASRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, processor: WhisperProcessor, lang_token_id: int):
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


class WhisperFineTuner(L.LightningModule):
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        lang_token_id: int,
        learning_rate: float,
        warmup_steps: int,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = processor.tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=["model", "processor"])

        for parameter in self.model.model.encoder.parameters():
            parameter.requires_grad = False
        # from_pretrained leaves the network in eval mode. Train the decoder
        # (dropout on) and keep the frozen encoder in eval.
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
            logging.getLogger("fine-tune").info(
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
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
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


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_dir = OUTPUT_DIR / args.model_name
    if local_rank == 0:
        setup_logging(output_dir / "train.log")
    log = logging.getLogger("fine-tune")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    device_ids = resolve_devices(args.devices)
    device_names = [name for name, index in NVIDIA_DEVICES.items() if index in device_ids]
    log.info("devices %s -> cuda %s", device_names or device_ids, device_ids)

    if not args.model_id:
        args.model_id = whisper_model()

    splits = load_splits()
    for name, frame in splits.items():
        log.info("%s clips: %d", name, len(frame))

    processor = WhisperProcessor.from_pretrained(args.model_id)
    lang_token = f"<|{args.language}|>"
    lang_token_id = processor.tokenizer.convert_tokens_to_ids(lang_token)
    unk_id = processor.tokenizer.unk_token_id
    if lang_token_id is None or lang_token_id == unk_id:
        raise SystemExit(f"{args.model_id} has no language token {lang_token}")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.use_cache = False

    def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = WhisperASRDataset(frame, processor, lang_token_id)

        def _seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=args.num_workers > 0,
        )

    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    dev_loader = make_loader(splits["dev"], args.eval_batch_size, shuffle=False)
    test_loader = make_loader(splits["test"], args.eval_batch_size, shuffle=False)

    module = WhisperFineTuner(
        model,
        processor=processor,
        lang_token_id=lang_token_id,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )
    frozen = sum(parameter.numel() for parameter in module.model.parameters() if not parameter.requires_grad)
    trainable = sum(parameter.numel() for parameter in module.model.parameters() if parameter.requires_grad)
    log.info("frozen encoder parameters: %d", frozen)
    log.info("trainable decoder parameters: %d", trainable)

    ckpt_dir = checkpoint_root() / args.model_name
    checkpoint_callback = BatchFractionCheckpoint(ckpt_dir, args.model_name, fraction=0.05)
    logger = CSVLogger(save_dir=str(OUTPUT_DIR), name=args.model_name)
    strategy = DDPStrategy(find_unused_parameters=False) if len(device_ids) > 1 else "auto"
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=device_ids,
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        default_root_dir=str(output_dir),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        enable_progress_bar=True,
    )

    ckpt_path = resume_checkpoint(ckpt_dir, args.resume)
    baseline_metrics: dict[str, float] = {}
    if ckpt_path is not None:
        log.info("resuming from %s", ckpt_path)
    elif args.baseline_eval:
        log.info("scoring the dev split before fine-tuning")
        trainer.validate(module, dataloaders=dev_loader)
        baseline_metrics = dict(module.val_metrics)

    trainer.fit(module, train_loader, dev_loader, ckpt_path=ckpt_path)
    trainer.test(module, dataloaders=test_loader)

    if trainer.is_global_zero:
        payload = {
            "model_id": args.model_id,
            "model_name": args.model_name,
            "language": args.language,
            "devices": device_names or device_ids,
            "epochs": args.max_epochs,
            "frozen": "encoder",
            "trainable": "decoder",
            "clips": {name: len(frame) for name, frame in splits.items()},
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


if __name__ == "__main__":
    main()
