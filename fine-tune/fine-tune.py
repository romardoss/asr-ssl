"""Fine-tune Sunbird/asr-whisper-51-african-languages on a subset of a
general_manifest-style CSV (see /data/kharin/extended_dataset), using
PyTorch Lightning.

Rows are selected via the `to_use` column, and split into train/validation
using the `subset` column: `train` + `validation` rows become the training
set, `test` rows become the validation set used during fine-tuning. Audio is
trimmed to the [start, end] seconds given in `vad_start_end` (when present)
before being fed to the model.

Checkpoints, the final model, and (if enabled) W&B logs all land under
<logdir>/<experiment>/.

Example:
    python scripts/finetune_whisper_lightning.py \
        --manifest /data/kharin/extended_dataset/luo_manifest_basefilt.csv \
        --language luo \
        --logdir logdirs \
        --experiment whisper-luo
"""

import argparse
import ast
import random
from pathlib import Path
from typing import Optional

import jiwer
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchaudio
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

DEFAULT_MODEL_ID = "Sunbird/asr-whisper-51-african-languages"

# Whisper's hard input limit -- longer clips get truncated by the feature
# extractor while their (untruncated) transcription would still be used as
# the label, so we drop anything longer than this instead of training on a
# mismatched audio/text pair.
MAX_AUDIO_SECONDS = 30.0

# Whisper's decoder max_target_positions: labels (prefix + text + eos) longer
# than this raise a ValueError from the model. Mixup-concatenated transcripts
# are the main way this gets hit, since two clips under MAX_AUDIO_SECONDS
# combined can still produce more text than either alone.
MAX_LABEL_TOKENS = 448

# From the model card: Whisper's native language tokens repurposed (and, for
# languages Whisper never shipped with, previously-unused tokens overwritten)
# to represent each of the 51 fine-tuned African languages.
LANGUAGE_TOKENS_WHISPER = {
    # Existing languges codes from Whisper
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    # Overwrite unused language tokens
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309
}

TRAIN_SUBSETS = {"train", "validation", "additional"}
VAL_SUBSETS = {"test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    parser.add_argument(
        "--language",
        required=True,
        help=f"ISO 639-3 language code to fine-tune, e.g. 'luo'. Must be one of: {sorted(LANGUAGE_TOKENS_WHISPER)}",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--logdir", default="logdirs", help="Base log directory.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name. Checkpoints/final model/W&B logs go under <logdir>/<experiment>/, "
        "and it's used as the W&B run name.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--devices", default="1", help="e.g. '1', '2', or '0,1'. Use 2+ to train with DDP across GPUs.")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default=None, help="If set, log metrics to this W&B project.")
    parser.add_argument(
        "--min-audio-length-train",
        type=float,
        default=0.0,
        help="Drop training clips shorter than this many seconds (measured on the real_duration "
        "column, i.e. the vad_start_end-trimmed length when available).",
    )
    parser.add_argument(
        "--min-audio-length-eval",
        type=float,
        default=0.0,
        help="Drop validation clips shorter than this many seconds (same measurement as "
        "--min-audio-length-train).",
    )
    parser.add_argument("--mixup", action="store_true", help="Enable train-only clip-concatenation augmentation.")
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.5,
        help="Probability of attempting mixup on a given training example.",
    )
    parser.add_argument(
        "--mixup-min-duration",
        type=float,
        default=0.0,
        help="Clips shorter than this many seconds always get a mixup partner, regardless of --mixup-prob.",
    )
    parser.add_argument(
        "--mixup-long-base-only",
        action="store_true",
        help="Only use clips with real_duration >= --mixup-min-duration as the base/anchor sample "
        "an epoch iterates over. Shorter clips are then only ever used as mixup partners glued "
        "onto a longer anchor, never trained on standalone.",
    )
    parser.add_argument(
        "--baseline-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a full validation pass on the un-tuned model before training, so val_loss/val_wer/"
        "val_cer/val_comp_metric are logged as a pre-fine-tuning benchmark. Use --no-baseline-eval to skip.",
    )
    # parser.add_argument(
    #     "--generate-max-new-tokens",
    #     type=int,
    #     default=225,
    #     help="Max new tokens when generating predictions for validation WER/CER.",
    # )
    return parser.parse_args()


def segment_duration(row: pd.Series) -> float:
    vad = row.get("vad_start_end")
    if isinstance(vad, str) and vad.strip():
        start_sec, end_sec = ast.literal_eval(vad)
        return end_sec - start_sec
    return row["duration"]


class WhisperASRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame, processor: WhisperProcessor, lang_token_id: int,
        do_mixup: bool = False,
        mixup_max_duration: Optional[float] = None, mixup_min_duration: Optional[float] = None,
        mixup_prob: float = 0.5, mixup_long_base_only: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.sampling_rate = processor.feature_extractor.sampling_rate

        transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.prefix_ids = [lang_token_id, transcribe_id, notimestamps_id]
        self.eos_token_id = self.tokenizer.eos_token_id

        self.do_mixup = do_mixup
        self.mixup_max_duration = mixup_max_duration
        self.mixup_min_duration = mixup_min_duration
        self.mixup_prob = mixup_prob

        if self.do_mixup:
            # Precomputed once so filtering mixup candidates doesn't retokenize the
            # whole dataframe on every __getitem__ call.
            self.df["num_tokens"] = self.df["transcription"].apply(
                lambda t: len(self.tokenizer(str(t), add_special_tokens=False).input_ids)
            )

        # anchor_indices controls which rows can be the "base"/anchor sample an
        # epoch iterates over. self.df itself (used for mixup partner search
        # below) always stays the full, unrestricted pool, so short clips can
        # still be picked as partners even when they can never be an anchor.
        if self.do_mixup and mixup_long_base_only:
            self.anchor_indices = self.df.index[self.df["real_duration"] >= mixup_min_duration].tolist()
        else:
            self.anchor_indices = self.df.index.tolist()

        print("Mixup Augmentation Settings:")
        print(f"  - Enabled: {self.do_mixup}")
        print(f"  - Minimum Duration: {self.mixup_min_duration}")
        print(f"  - Maximum Duration: {self.mixup_max_duration}")
        print(f"  - Probability: {self.mixup_prob}")
        print(f"  - Long-base-only anchors: {mixup_long_base_only} ({len(self.anchor_indices)}/{len(self.df)} rows)")

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _get_sample(self, idx: int):
        row = self.df.iloc[idx]
        
        waveform, sr = torchaudio.load(row["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            sr = self.sampling_rate

        vad = row.get("vad_start_end")
        if isinstance(vad, str) and vad.strip():
            start_sec, end_sec = ast.literal_eval(vad)
            waveform = waveform[:, int(start_sec * sr) : int(end_sec * sr)]

        audio = waveform.squeeze(0).numpy()

        return audio, sr, row["transcription"]

    @staticmethod
    def _merge_transcriptions(transcriptions: list[str]) -> str:
        assert len(transcriptions) == 2, f"Expected 2 transcriptions for mixup, got {len(transcriptions)}"
        if transcriptions[0].endswith("."):
            return transcriptions[0] + " " + transcriptions[1]
        else:
            return transcriptions[0] + ". " + transcriptions[1]

    def __getitem__(self, idx: int) -> dict:

        real_idx = self.anchor_indices[idx]
        audio, sr, transcription = self._get_sample(real_idx)
        original_audio, original_transcription = audio, transcription

        if self.do_mixup:
            if len(audio) / sr < self.mixup_min_duration or np.random.binomial(n=1, p=self.mixup_prob):
                additional_duration = self.mixup_max_duration - len(audio) / sr
                # Reserve room for the [lang, transcribe, notimestamps] prefix and the
                # trailing eos token, and only consider a partner whose own token
                # count can't push the combined label past Whisper's decoder limit.
                reserved_tokens = len(self.prefix_ids) + 1
                current_num_tokens = self.df.loc[real_idx, "num_tokens"]
                max_partner_tokens = MAX_LABEL_TOKENS - reserved_tokens - current_num_tokens
                possible_samples = self.df[
                    (self.df["real_duration"] <= additional_duration)
                    & (self.df["num_tokens"] <= max_partner_tokens)
                    & (self.df.index != real_idx)
                ]
                if not possible_samples.empty:
                    mixup_idx = possible_samples.sample(n=1).index[0]
                    mixup_audio, mixup_sr, mixup_transcription = self._get_sample(mixup_idx)
                    assert mixup_sr == sr, f"Mixup sample {mixup_idx} has different sampling rate {mixup_sr} != {sr}"
                    # Independent fair coin for ordering -- this used to reuse self.mixup_prob,
                    # which meant "how often to mixup" also silently biased "which clip comes first".
                    # print("Original audio length:", len(audio) / sr, "s; " "Original transcription:", transcription)
                    if np.random.binomial(n=1, p=0.5):
                        audio = np.concatenate((audio, mixup_audio))
                        transcription = self._merge_transcriptions([transcription, mixup_transcription])
                    else:
                        audio = np.concatenate((mixup_audio, audio))
                        transcription = self._merge_transcriptions([mixup_transcription, transcription])
                    # print("Mixup audio length:", len(audio) / sr, "s; " "Mixup transcription:", transcription)


        extracted = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
        )

        text_ids = self.tokenizer(str(transcription), add_special_tokens=False).input_ids
        if len(text_ids) + len(self.prefix_ids) + 1 > MAX_LABEL_TOKENS:
            # Rare fallback: the per-clip token-count pre-filter bounds the common
            # case, but retokenizing the *merged* string can add a token or two at
            # the join point (BPE merge boundary). Revert to the unmixed sample
            # rather than exceed Whisper's max_target_positions.
            audio, transcription = original_audio, original_transcription
            extracted = self.feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
            )
            text_ids = self.tokenizer(str(transcription), add_special_tokens=False).input_ids

        labels = self.prefix_ids + text_ids + [self.eos_token_id]

        return {
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
            # Kept only for occasionally logging a qualitative validation sample
            # (the exact audio the model saw, post VAD-trim/resample).
            "waveform": audio,
            "sampling_rate": sr,
        }


def collate_fn(features: list[dict], pad_token_id: int) -> dict:
    input_features = torch.stack([f["input_features"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    max_label_len = max(len(f["labels"]) for f in features)
    labels = torch.full((len(features), max_label_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        labels[i, : len(f["labels"])] = torch.tensor(f["labels"], dtype=torch.long)
    return {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "labels": labels,
        "waveforms": [f["waveform"] for f in features],
        "sampling_rate": features[0]["sampling_rate"],
    }


class WhisperFineTuner(L.LightningModule):
    def __init__(
        self,
        model,
        processor: WhisperProcessor,
        lang_token_id: int,
        learning_rate: float,
        warmup_steps: int,
        # generate_max_new_tokens: int = 225,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = processor.tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        # self.generate_max_new_tokens = generate_max_new_tokens

        transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        # Prompt-based prefix (decoder_input_ids) rather than `forced_decoder_ids`:
        # the latter is deprecated and also makes generate() build a second,
        # conflicting SuppressTokensAtBeginLogitsProcessor on top of the one the
        # model's own generation_config already defines.
        self.decoder_prompt_ids = [
            model.config.decoder_start_token_id,
            lang_token_id,
            transcribe_id,
            notimestamps_id,
        ]

        self.val_references: list[str] = []
        self.val_hypotheses: list[str] = []
        self.val_sample_text_rows: list[list] = []

    @staticmethod
    def _model_inputs(batch: dict) -> dict:
        # `batch` also carries "waveforms"/"sampling_rate" (only needed for
        # qualitative logging), which the model's forward() doesn't accept.
        return {
            "input_features": batch["input_features"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def training_step(self, batch, batch_idx):
        loss = self.model(**self._model_inputs(batch)).loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**self._model_inputs(batch)).loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        batch_size = batch["input_features"].shape[0]
        decoder_input_ids = torch.tensor(
            [self.decoder_prompt_ids] * batch_size, device=self.device
        )
        predicted_ids = self.model.generate(
            batch["input_features"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            # max_new_tokens=self.generate_max_new_tokens,
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

        self.val_hypotheses.extend(hypotheses)
        self.val_references.extend(references)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self._log_val_sample(
                spec=batch["input_features"][0],
                waveform=batch["waveforms"][0],
                sampling_rate=batch["sampling_rate"],
                gt_text=references[0],
                pred_text=hypotheses[0],
            )

        return loss

    def _log_val_sample(self, spec, waveform, sampling_rate, gt_text, pred_text):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(spec.float().cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        fig.tight_layout()

        self.logger.experiment.log(
            {
                "val_sample/spectrogram": wandb.Image(fig),
                "val_sample/audio": wandb.Audio(waveform, sample_rate=sampling_rate),
            },
            step=self.global_step,
        )
        plt.close(fig)

        # gt/pred text logged as a growing table instead of flat keys, since a
        # table lets you scroll through every epoch's sample side by side.
        # wandb.Table can't have rows added after it's been logged once, so we
        # rebuild it from the full accumulated history each time and re-log it
        # under the same key.
        self.val_sample_text_rows.append([self.current_epoch, self.global_step, gt_text, pred_text])
        table = wandb.Table(
            columns=["epoch", "step", "gt_text", "pred_text"], data=self.val_sample_text_rows
        )
        self.logger.experiment.log({"val_sample_text": table}, step=self.global_step)

    def on_validation_epoch_end(self):
        if self.val_references:
            wer = jiwer.wer(self.val_references, self.val_hypotheses)
            cer = jiwer.cer(self.val_references, self.val_hypotheses)
            self.log("val_wer", wer, prog_bar=True, sync_dist=True)
            self.log("val_cer", cer, prog_bar=True, sync_dist=True)
            self.log("val_comp_metric", 1 - (cer + wer) / 2, prog_bar=True, sync_dist=True)
        self.val_references.clear()
        self.val_hypotheses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # trainer.estimated_stepping_batches accounts for accumulate_grad_batches and
        # the number of devices/dataset-sharding under DDP, unlike a manually
        # precomputed step count, which would be wrong once training is split
        # across multiple GPUs.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    # Gradient checkpointing + gradient accumulation intentionally keep an
    # autograd graph reference alive across steps; this is the mismatch
    # PyTorch is warning about, and it's explicit about this being the
    # suppression mechanism for exactly that intentional case.
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    # # These are harmless artifacts of combining a custom decoder_input_ids prompt
    # # with the checkpoint's own generation_config (max_length/begin_suppress_tokens);
    # # the docs confirm our explicit args always take precedence, so this just
    # # quiets the noise rather than changing any actual generation behavior.
    # hf_logging.set_verbosity_error()

    if args.language not in LANGUAGE_TOKENS_WHISPER:
        raise SystemExit(f"Unknown language {args.language!r}. Known: {sorted(LANGUAGE_TOKENS_WHISPER)}")
    lang_token_id = LANGUAGE_TOKENS_WHISPER[args.language]

    if args.mixup and args.mixup_min_duration > MAX_AUDIO_SECONDS:
        raise SystemExit(
            f"--mixup-min-duration ({args.mixup_min_duration}) cannot exceed "
            f"MAX_AUDIO_SECONDS ({MAX_AUDIO_SECONDS})"
        )

    df = pd.read_csv(args.manifest)
    df = df[df["language"] == args.language]
    df["real_duration"] = df.apply(segment_duration, axis=1)

    too_long = df["real_duration"] > MAX_AUDIO_SECONDS
    if too_long.any():
        print(f"Dropping {too_long.sum()} row(s) longer than {MAX_AUDIO_SECONDS}s (Whisper's input limit)")
    df = df[~too_long]

    train_df = df[df["subset"].isin(TRAIN_SUBSETS)]
    val_df = df[df["subset"].isin(VAL_SUBSETS)]

    # Just because "duration"
    train_too_short = train_df["duration"] < args.min_audio_length_train
    if train_too_short.any():
        print(f"Dropping {train_too_short.sum()} train row(s) shorter than {args.min_audio_length_train}s")
    train_df = train_df[~train_too_short]

    val_too_short = val_df["duration"] < args.min_audio_length_eval
    if val_too_short.any():
        print(f"Dropping {val_too_short.sum()} val row(s) shorter than {args.min_audio_length_eval}s")
    val_df = val_df[~val_too_short]

    print(f"Train rows: {len(train_df)} (subset in {TRAIN_SUBSETS}); Val rows: {len(val_df)} (subset in {VAL_SUBSETS})")

    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    # # The checkpoint ships a default max_length=200, which conflicts with the
    # # max_new_tokens we pass at generation time; clear it so only max_new_tokens applies.
    # model.generation_config.max_length = None

    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = WhisperASRDataset(
        # train_df.iloc[:32],
        train_df,
        processor,
        lang_token_id,
        do_mixup=args.mixup,
        mixup_prob=args.mixup_prob,
        mixup_min_duration=args.mixup_min_duration,
        mixup_max_duration=MAX_AUDIO_SECONDS,
        mixup_long_base_only=args.mixup_long_base_only,
    )
    val_dataset = WhisperASRDataset(val_df, processor, lang_token_id)
    pad_token_id = processor.tokenizer.pad_token_id

    def _seed_worker(worker_id: int) -> None:
        # DataLoader workers are forked and otherwise inherit numpy's/random's
        # global RNG state identically, which would make mixup's "random"
        # partner/order/trigger choices correlated across workers.
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        collate_fn=lambda features: collate_fn(features, pad_token_id),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda features: collate_fn(features, pad_token_id),
    )

    module = WhisperFineTuner(
        model,
        processor=processor,
        lang_token_id=lang_token_id,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # generate_max_new_tokens=args.generate_max_new_tokens,
    )

    output_dir = Path(args.logdir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch}-{val_comp_metric:.4f}",
        monitor="val_comp_metric",
        mode="max",
        save_top_k=1,
        save_last=True
    )

    logger = True
    if args.wandb_project:
        logger = WandbLogger(
            save_dir=str(output_dir),
            project=args.wandb_project,
            name=args.experiment,
            id=None,  # placeholder so a run can be resumed by setting this later
            log_model=False,
        )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        devices=args.devices,
        strategy=args.strategy,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        default_root_dir=str(output_dir),
        logger=logger,
        # Lightning's built-in sanity check only runs a couple of batches and,
        # more importantly, never forwards its metrics to the real logger
        # (W&B/CSV) -- so it can't serve as a benchmark. We run a full,
        # properly-logged validation pass below instead and skip the sanity
        # check entirely.
        num_sanity_val_steps=0,
        # num_sanity_val_steps=2,
        # enable_progress_bar=False
    )

    if args.baseline_eval:
        print("Running full validation on the un-tuned model for a pre-fine-tuning benchmark...")
        trainer.validate(module, dataloaders=val_loader)

    trainer.fit(module, train_loader, val_loader)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
