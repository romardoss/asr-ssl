"""Score a Whisper model on the Common Voice test split.

The model is a checkpoint or directory path, or ``model.id`` from
``config.yaml`` when ``--model`` is omitted. A ``.ckpt`` file is a fine-tune
checkpoint loaded on top of ``model.id``. GPUs, language, and the batch size
come from that same file.

Results and plots are written to ``evaluation/testing-<model>-XX/``.

    python evaluation/evaluate.py
    python evaluation/evaluate.py --model /path/to/last.ckpt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import queue
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
FINE_TUNE_PATH = ROOT / "fine-tune" / "fine-tune.py"

plt.rcParams.update(
    {
        "figure.figsize": (8, 5.5),
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    }
)


def _load_fine_tune():
    # Importing fine-tune pins CUDA_VISIBLE_DEVICES from config.yaml
    # before torch initializes CUDA.
    spec = importlib.util.spec_from_file_location("whisper_fine_tune", FINE_TUNE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {FINE_TUNE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ft = _load_fine_tune()

import jiwer
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint (.ckpt) or model directory. Default: model.id from config.yaml.",
    )
    return parser.parse_args()


def resolve_source(raw: str | None) -> tuple[str, str]:
    """Return ``(kind, source)`` where kind is hf, ckpt, or dir."""
    if raw is None:
        return "hf", ft.whisper_model()
    path = Path(raw).expanduser()
    if path.is_file():
        if path.suffix != ".ckpt":
            raise SystemExit(f"Model file must be a .ckpt checkpoint: {path}")
        return "ckpt", str(path.resolve())
    if path.is_dir():
        return "dir", str(path.resolve())
    if "/" in raw or raw.endswith(".ckpt"):
        raise SystemExit(f"Model path not found: {raw}")
    return "hf", raw


def model_slug(kind: str, source: str) -> str:
    if kind == "hf":
        name = source.rstrip("/").split("/")[-1]
    elif kind == "ckpt":
        name = Path(source).stem
    else:
        name = Path(source).name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "model"


def next_run_dir(slug: str) -> Path:
    prefix = f"testing-{slug}-"
    numbers = []
    for path in EVALUATION_DIR.glob(f"{prefix}*"):
        suffix = path.name.removeprefix(prefix)
        if path.is_dir() and suffix.isdigit():
            numbers.append(int(suffix))
    number = max(numbers) + 1 if numbers else 1
    return EVALUATION_DIR / f"{prefix}{number:02d}"


def load_whisper(kind: str, source: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if kind == "ckpt":
        base = ft.whisper_model()
        processor = WhisperProcessor.from_pretrained(base)
        model = WhisperForConditionalGeneration.from_pretrained(base)
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


def language_token_id(processor, language: str) -> int:
    token = f"<|{language}|>"
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == processor.tokenizer.unk_token_id:
        raise SystemExit(f"Model has no language token {token}")
    return token_id


def evaluate_shard(
    frame: pd.DataFrame,
    kind: str,
    source: str,
    language: str,
    batch_size: int,
    visible_index: int,
    log_path: Path,
) -> tuple[list[str], list[str], float, int]:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
    device = torch.device("cuda", visible_index)
    torch.cuda.set_device(device)
    model, processor = load_whisper(kind, source)
    token_id = language_token_id(processor, language)
    language_name = processor.tokenizer.convert_ids_to_tokens(token_id).strip("<|>")
    model.config.use_cache = False
    model.generation_config.language = language_name
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    model.eval()

    dataset = ft.WhisperASRDataset(frame, processor, token_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ft.collate_fn,
        pin_memory=True,
    )
    references: list[str] = []
    hypotheses: list[str] = []
    loss_sum = 0.0
    batches = 0
    for batch_index, batch in enumerate(loader, start=1):
        features = batch["input_features"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_features=features, attention_mask=mask, labels=labels).loss
            predicted_ids = model.generate(
                features,
                attention_mask=mask,
                language=language_name,
                task="transcribe",
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        loss_sum += float(loss.detach().cpu())
        batches += 1
        label_ids = labels.clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hypotheses.extend(
            processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )
        references.extend(
            processor.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )
        if batch_index == 1 or batch_index % 20 == 0:
            logging.getLogger("testing").info(
                "cuda:%d batch %d/%d", visible_index, batch_index, len(loader)
            )
    mean_loss = loss_sum / batches if batches else 0.0
    return references, hypotheses, mean_loss, batches


def _worker(
    visible_index: int,
    frame,
    kind: str,
    source: str,
    language: str,
    batch_size: int,
    log_path: str,
    queue,
) -> None:
    try:
        references, hypotheses, loss, batches = evaluate_shard(
            frame, kind, source, language, batch_size, visible_index, Path(log_path)
        )
        queue.put(("ok", visible_index, references, hypotheses, loss, batches))
    except (Exception, SystemExit) as exc:
        queue.put(("err", visible_index, f"{type(exc).__name__}: {exc}"))


def run_evaluation(
    frame: pd.DataFrame,
    kind: str,
    source: str,
    language: str,
    batch_size: int,
    log_path: Path,
) -> tuple[list[str], list[str], float]:
    devices = list(range(len(ft.PHYSICAL_GPUS)))
    if len(devices) == 1:
        references, hypotheses, loss, _batches = evaluate_shard(
            frame, kind, source, language, batch_size, devices[0], log_path
        )
        return references, hypotheses, loss

    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = []
    shards = [frame.iloc[index:: len(devices)].reset_index(drop=True) for index in devices]
    for index, shard in zip(devices, shards):
        process = context.Process(
            target=_worker,
            args=(index, shard, kind, source, language, batch_size, str(log_path), queue),
        )
        process.start()
        processes.append(process)

    messages = []
    while len(messages) < len(processes):
        try:
            messages.append(queue.get(timeout=1))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break
    for process in processes:
        process.join()
    failed = [process for process in processes if process.exitcode not in (0, None)]
    errors = [message for message in messages if message[0] != "ok"]
    if errors or failed:
        detail = "; ".join(message[-1] for message in errors) or "worker failed"
        raise SystemExit(detail)

    by_index = {message[1]: message for message in messages}
    references: list[str] = []
    hypotheses: list[str] = []
    loss_sum = 0.0
    batch_count = 0
    slots: list[tuple[str, str] | None] = [None] * len(frame)
    for index in devices:
        _status, _index, refs, hyps, loss, batches = by_index[index]
        positions = range(index, len(frame), len(devices))
        for position, reference, hypothesis in zip(positions, refs, hyps):
            slots[position] = (reference, hypothesis)
        loss_sum += loss * batches
        batch_count += batches
    if any(slot is None for slot in slots):
        raise SystemExit("Evaluation did not return a transcript for every clip.")
    for reference, hypothesis in slots:
        references.append(reference)
        hypotheses.append(hypothesis)
    mean_loss = loss_sum / batch_count if batch_count else 0.0
    return references, hypotheses, mean_loss


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_plots(run_dir: Path, metrics: dict[str, float], references: list[str], hypotheses: list[str]) -> list[Path]:
    plots = run_dir / "plots"
    written: list[Path] = []

    rate_labels = []
    rate_values = []
    for label, key in (("WER", "wer"), ("CER", "cer"), ("MER", "mer")):
        if key not in metrics:
            continue
        rate_labels.append(label)
        rate_values.append(metrics[key] * 100)
    if rate_labels:
        figure, axis = plt.subplots()
        bars = axis.bar(
            rate_labels,
            rate_values,
            color=["#9b2226", "#1d4e89", "#2a9d8f"][: len(rate_labels)],
            width=0.6,
        )
        for bar, value in zip(bars, rate_values):
            axis.annotate(
                f"{value:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=12,
            )
        axis.set_title("Test set error rates")
        axis.set_ylabel("error rate (%)")
        axis.set_ylim(0, max(max(rate_values) * 1.25, 1))
        path = plots / "error_rates.png"
        save_figure(figure, path)
        written.append(path)

    edit_labels = []
    edit_values = []
    for label, key in (("substitutions", "substitutions"), ("deletions", "deletions"), ("insertions", "insertions")):
        if key not in metrics:
            continue
        edit_labels.append(label)
        edit_values.append(metrics[key])
    if edit_labels:
        figure, axis = plt.subplots()
        bars = axis.bar(
            edit_labels,
            edit_values,
            color=["#9b2226", "#e09f3e", "#1d4e89"][: len(edit_labels)],
            width=0.6,
        )
        for bar, value in zip(bars, edit_values):
            axis.annotate(
                f"{value:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=12,
            )
        axis.set_title("Test set word edits")
        axis.set_ylabel("count")
        axis.set_ylim(0, max(max(edit_values) * 1.25, 1))
        path = plots / "edit_counts.png"
        save_figure(figure, path)
        written.append(path)

    scores = []
    for reference, hypothesis in zip(references, hypotheses):
        normalized_ref = ft.normalize_text(reference)
        if not normalized_ref:
            continue
        scores.append(jiwer.wer(normalized_ref, ft.normalize_text(hypothesis)) * 100)
    if scores:
        figure, axis = plt.subplots()
        axis.hist(scores, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200], color="#1f4e79")
        axis.set_title("Utterance WER")
        axis.set_xlabel("WER (%)")
        axis.set_ylabel("clips")
        path = plots / "utterance_wer.png"
        save_figure(figure, path)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    kind, source = resolve_source(args.model)
    language = ft.SETTINGS.language
    batch_size = ft.SETTINGS.testing_batch_size
    slug = model_slug(kind, source)
    run_dir = next_run_dir(slug)
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "test.log"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("testing")
    device_names = [f"nvidia{index}" for index in ft.PHYSICAL_GPUS]
    log.info(
        "model=%s kind=%s gpus=%s CUDA_VISIBLE_DEVICES=%s testing_batch=%d",
        source,
        kind,
        device_names,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        batch_size,
    )

    frame = ft.load_splits()["test"].reset_index(drop=True)
    log.info("test clips: %d", len(frame))
    torch.set_float32_matmul_precision("high")
    references, hypotheses, loss = run_evaluation(
        frame, kind, source, language, batch_size, run_dir / "test.log"
    )
    metrics = ft.corpus_metrics(references, hypotheses)
    predictions = pd.DataFrame(
        {
            "audio_path": frame["audio_path"].tolist(),
            "reference": references,
            "hypothesis": hypotheses,
        }
    )
    predictions_path = run_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    payload = {
        "model": source,
        "kind": kind,
        "base_model": ft.whisper_model() if kind == "ckpt" else source,
        "language": language,
        "split": "test",
        "clips": len(frame),
        "gpus": device_names,
        "eval_batch_size": batch_size,
        "loss": loss,
        "metrics": metrics,
    }
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written = write_plots(run_dir, metrics, references, hypotheses)
    log.info("loss %.4f metrics %s", loss, metrics)
    log.info("wrote %s", results_path)
    log.info("wrote %s", predictions_path)
    for path in written:
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
