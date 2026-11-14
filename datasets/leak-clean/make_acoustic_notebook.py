"""Generate ``review_acoustic.ipynb``: find re-encoded copies of the same recording.

SHA-256 only finds byte-identical files. The same recording can also sit in two
datasets as two different encodings (for example a 50 KB and a 200 KB mp3), and
then the bytes differ while the sound is the same. The notebook looks for those:

1. candidates: eval clips and train clips with the same normalized text or the
   same file name (the only pairs worth comparing);
2. decode both clips to 16 kHz mono with ffmpeg and drop pairs whose durations
   differ too much;
3. score = best Pearson correlation of the log-mel spectrograms over small time
   shifts (codecs add silence), after removing each mel bin's mean so that only
   the time pattern counts, not the speaker-independent average spectrum.

The notebook only searches and shows pairs to listen to. It cleans nothing. When
the threshold is confirmed by ear, one cell writes ``acoustic_matches.csv`` for
the preprocessing pipeline.

Needs ``ffmpeg`` on the machine that has the audio. Run from any folder:

    python datasets/leak-clean/make_acoustic_notebook.py
    jupyter lab datasets/leak-clean/review_acoustic.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

from make_review_notebook import AUDIO  # load_audio / mime_of / player, shared with review_leaks.ipynb

NOTEBOOK = Path(__file__).resolve().with_name("review_acoustic.ipynb")

INTRO = """\
# Акустичні дублікати: той самий запис в іншому кодуванні

`find_audio_matches.py` знаходить лише однакові байти (SHA-256) і однакові імена файлів. Ручне
прослуховування показало, що частина пар `cv10_test` / `common_voice` — той самий запис у двох кодуваннях
(наприклад 50 КБ і 200 КБ): звучить однаково, байти різні. Орієнтовно це ~15 % пар з однаковим текстом.

Алгоритм:

1. Кандидати: eval-клип і train-клип з однаковим `text_norm` або однаковим іменем файлу.
2. Обидва клипи декодуються через `ffmpeg` у 16 кГц mono. Пари з різницею тривалості понад `MAX_DURATION_DIFF` відкидаються.
3. `score` — найкраща кореляція Пірсона між log-mel спектрограмами при зсувах до `MAX_SHIFT` секунд
   (кодеки додають тишу на початку), після віднімання середнього по кожному mel-діапазону.
   Той самий запис дає близько 1, різні диктори з тим самим текстом — помітно менше.

Ноутбук нічого не чистить. Він шукає пари й показує їх для прослуховування, відсортовані за `score` від
найбільшого. Гортайте `PAGE`, доки пари не перестануть бути тим самим записом: це й буде поріг. Задайте його
в `THRESHOLD`, і остання клітинка запише `acoustic_matches.csv` для препроцесингу.

Запускати треба на машині з аудіо й `ffmpeg`. Перед збереженням чи комітом очистіть виводи.
"""

SETUP = '''\
import base64
import html
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from IPython.display import HTML, display

# Folder with the leak-clean scripts: found from the notebook's folder upwards.
LEAK_DIR = next(
    folder / name
    for folder in [Path.cwd(), *Path.cwd().parents]
    for name in ("", "datasets/leak-clean")
    if (folder / name / "leak_common.py").is_file()
)
sys.path.insert(0, str(LEAK_DIR))

from find_audio_matches import load_compared
from leak_common import output_dir

FOLDER = Path(os.environ["LEAK_OUTPUT"]) if os.environ.get("LEAK_OUTPUT") else output_dir(None)
print("scripts:", LEAK_DIR)
print("metadata:", FOLDER)
subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)  # fails early when ffmpeg is missing
'''

PARAMS = '''\
# What to compare. Re-run this cell and everything below after changing it.
EVAL_DATASETS = ["cv10_test"]       # datasets whose dev/test clips are checked
EVAL_SPLITS = ["dev", "test"]
TRAIN_DATASETS = ["common_voice"]   # datasets whose train clips they are compared with
TRAIN_SPLITS = ["train"]
CANDIDATE_KEYS = ("text", "name")   # pair clips by the same normalized text and/or the same file name
MIN_WORDS = 1                       # shortest text used for "text" candidates
MAX_PAIRS = 2000                    # random sample of candidate pairs to score; None scores all of them
MAX_DURATION_DIFF = 0.5             # seconds; pairs that differ more are not the same recording
MAX_SHIFT = 0.3                     # seconds; the largest time shift tried when aligning two clips
THRESHOLD = 0.8                     # score at or above this counts as the same recording (find it by ear)
SEED = 0

# What to show. Change it and re-run the "Прослуховування" cell.
PAGE = 0
PAGE_SIZE = 5                       # pairs per page
'''

CANDIDATES = '''\
metadata = load_compared(FOLDER)
COLUMNS = ["dataset", "split", "id", "audio_name", "audio_sha256", "location", "text_norm", "words"]


def side(datasets, splits, suffix):
    clips = metadata[metadata["dataset"].isin(datasets) & metadata["split"].isin(splits)]
    return clips[COLUMNS].add_suffix("_" + suffix)


eval_clips = side(EVAL_DATASETS, EVAL_SPLITS, "eval")
train_clips = side(TRAIN_DATASETS, TRAIN_SPLITS, "train")
print(f"eval clips: {len(eval_clips)}, train clips: {len(train_clips)}")

found = []
if "text" in CANDIDATE_KEYS:
    left = eval_clips[(eval_clips["text_norm_eval"] != "") & (eval_clips["words_eval"] >= MIN_WORDS)]
    pairs_text = left.merge(train_clips, left_on="text_norm_eval", right_on="text_norm_train")
    pairs_text["key"] = "text"
    found.append(pairs_text)
if "name" in CANDIDATE_KEYS:
    left = eval_clips[eval_clips["audio_name_eval"] != ""]
    pairs_name = left.merge(train_clips, left_on="audio_name_eval", right_on="audio_name_train")
    pairs_name["key"] = "name"
    found.append(pairs_name)

pairs = pd.concat(found, ignore_index=True)
# A pair found by both keys is kept once, marked "name+text".
pair_columns = ["dataset_eval", "split_eval", "id_eval", "dataset_train", "split_train", "id_train"]
pairs["key"] = pairs.groupby(pair_columns)["key"].transform(lambda keys: "+".join(sorted(set(keys))))
pairs = pairs.drop_duplicates(pair_columns).reset_index(drop=True)
pairs["same_sha"] = (pairs["audio_sha256_eval"] != "") & (pairs["audio_sha256_eval"] == pairs["audio_sha256_train"])

print(f"candidate pairs: {len(pairs)} ({pairs['same_sha'].sum()} with identical SHA-256, i.e. known leaks)")
print(pairs.groupby("key").size().to_string())
if MAX_PAIRS is not None and len(pairs) > MAX_PAIRS:
    pairs = pairs.sample(MAX_PAIRS, random_state=SEED).reset_index(drop=True)
    print(f"scoring a random sample of {MAX_PAIRS} pairs (MAX_PAIRS)")
'''

FEATURES = '''\
SAMPLE_RATE = 16000
N_FFT = 400   # 25 ms
HOP = 160     # 10 ms
N_MELS = 40


def mel_filterbank() -> np.ndarray:
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    edges = to_hz(np.linspace(to_mel(50), to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    bank = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
    for index in range(N_MELS):
        low, mid, high = edges[index : index + 3]
        bank[index] = np.clip(np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)), 0, None)
    return bank


MEL_BANK = mel_filterbank()
WINDOW = np.hanning(N_FFT).astype(np.float32)


def decode(data: bytes) -> np.ndarray:
    """Encoded audio -> float32 mono 16 kHz waveform, through ffmpeg."""
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(run.stdout, dtype=np.int16).astype(np.float32) / 32768


def log_mel(wave: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram, frames x N_MELS."""
    frames = np.lib.stride_tricks.sliding_window_view(wave, N_FFT)[::HOP] * WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(power @ MEL_BANK.T + 1e-10).astype(np.float32)


def similarity(first: np.ndarray, second: np.ndarray, max_shift: int) -> tuple[float, int]:
    """Best Pearson correlation of two log-mel matrices over time shifts, and that shift in frames."""
    best, best_shift = -1.0, 0
    for shift in range(-max_shift, max_shift + 1):
        x = first[shift:] if shift >= 0 else first
        y = second if shift >= 0 else second[-shift:]
        length = min(len(x), len(y))
        if length < 50:  # under 0.5 s of overlap says nothing
            continue
        x = x[:length] - x[:length].mean(axis=0)  # drop each mel bin's mean: keep only the time pattern
        y = y[:length] - y[:length].mean(axis=0)
        denominator = np.sqrt((x * x).sum() * (y * y).sum())
        if denominator > 0:
            score = float((x * y).sum() / denominator)
            if score > best:
                best, best_shift = score, shift
    return best, best_shift
'''

SCORE = '''\
cache = globals().get("cache", {})  # (dataset, location) -> (bytes, duration, log-mel); kept when this cell is re-run


def clip_features(dataset: str, location: str):
    if (dataset, location) not in cache:
        data = load_audio(dataset, location)
        wave = decode(data)
        cache[(dataset, location)] = (len(data), len(wave) / SAMPLE_RATE, log_mel(wave) if len(wave) >= N_FFT else None)
    return cache[(dataset, location)]


max_shift = round(MAX_SHIFT * SAMPLE_RATE / HOP)
rows = []
for number, pair in enumerate(pairs.itertuples(index=False), 1):
    try:
        size_eval, duration_eval, mel_eval = clip_features(pair.dataset_eval, pair.location_eval)
        size_train, duration_train, mel_train = clip_features(pair.dataset_train, pair.location_train)
        score, shift = float("nan"), 0
        if mel_eval is not None and mel_train is not None and abs(duration_eval - duration_train) <= MAX_DURATION_DIFF:
            score, shift = similarity(mel_eval, mel_train, max_shift)
        rows.append((score, shift * HOP / SAMPLE_RATE * 1000, size_eval, size_train, duration_eval, duration_train, ""))
    except Exception as error:  # a missing or broken file should not stop the run
        rows.append((float("nan"), 0, 0, 0, 0.0, 0.0, str(error)[:80]))
    if number % 200 == 0 or number == len(pairs):
        print(f"{number}/{len(pairs)} pairs", flush=True)

scored = pairs.join(
    pd.DataFrame(rows, columns=["score", "shift_ms", "bytes_eval", "bytes_train", "duration_eval", "duration_train", "error"])
)
scored["duration_diff"] = (scored["duration_eval"] - scored["duration_train"]).abs()
print(f"errors: {(scored['error'] != '').sum()}, skipped by duration: {(scored['score'].isna() & (scored['error'] == '')).sum()}")

compared = scored.dropna(subset=["score"])
print(f"compared: {len(compared)} pairs; score >= {THRESHOLD}: {(compared['score'] >= THRESHOLD).sum()}")
print("pairs with identical SHA-256 (must score high):")
print(compared[compared["same_sha"]]["score"].describe().to_string() if compared["same_sha"].any() else "none in this sample")
print("score distribution:")
print(pd.cut(compared["score"], [-1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]).value_counts().sort_index().to_string())
'''

UI = '''\
# All compared pairs, best score first. Page through them with PAGE to find the score where "same recording" stops.
shown = compared.sort_values("score", ascending=False).reset_index(drop=True)


def clip_row(role: str, pair, suffix: str) -> str:
    return (
        "<tr>"
        f"<td><b>{role}</b></td>"
        f"<td>{html.escape(pair['dataset_' + suffix] + '/' + pair['split_' + suffix])}</td>"
        f"<td><small>{html.escape(str(pair['id_' + suffix]))}</small></td>"
        f"<td>{html.escape(str(pair['text_norm_' + suffix]))}</td>"
        f"<td>{player(pair['dataset_' + suffix], pair['location_' + suffix])}</td>"
        "</tr>"
    )


def pair_html(index: int, pair) -> str:
    note = (
        f"#{index}: score {pair['score']:.3f}, shift {pair['shift_ms']:.0f} ms, "
        f"duration {pair['duration_eval']:.2f} s / {pair['duration_train']:.2f} s, key {pair['key']}"
    )
    if pair["same_sha"]:
        note += ", <b>identical SHA-256</b>"
    return (
        '<div style="border:1px solid #8885;padding:6px;margin:6px 0">'
        f"<p>{note}</p>"
        f"<table>{clip_row('eval', pair, 'eval')}{clip_row('train', pair, 'train')}</table></div>"
    )


pages = max(1, -(-len(shown) // PAGE_SIZE))
start = PAGE * PAGE_SIZE
print(f"page {PAGE} (valid: 0-{pages - 1}), pairs {start}-{min(start + PAGE_SIZE, len(shown)) - 1} of {len(shown)}")
for index in range(start, min(start + PAGE_SIZE, len(shown))):
    display(HTML(pair_html(index, shown.iloc[index])))
'''

EXPORT = '''\
# Run this only when you have found the threshold by ear and set THRESHOLD to it.
# It writes the findings for the preprocessing pipeline: every eval/train pair scored >= THRESHOLD.
if MAX_PAIRS is not None:
    raise SystemExit("MAX_PAIRS is not None: only a sample was scored. Set MAX_PAIRS = None and re-run from the 'Кандидати' cell.")

matches = compared[compared["score"] >= THRESHOLD].copy()
columns = [
    "dataset_eval", "split_eval", "id_eval", "dataset_train", "split_train", "id_train",
    "score", "shift_ms", "duration_diff", "bytes_eval", "bytes_train", "key", "same_sha",
]
matches_path = FOLDER / "acoustic_matches.csv"
matches.sort_values("score", ascending=False)[columns].to_csv(matches_path, index=False)
print(f"wrote {matches_path} ({len(matches)} pairs, threshold {THRESHOLD})")
print(matches.groupby(["dataset_eval", "split_eval", "dataset_train", "split_train"]).size().to_string())
'''

def cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def main() -> None:
    notebook = {
        "cells": [
            cell("markdown", INTRO),
            cell("code", SETUP),
            cell("markdown", "## Параметри"),
            cell("code", PARAMS),
            cell("markdown", "## Кандидати: однаковий текст або ім'я файлу"),
            cell("code", CANDIDATES),
            cell("markdown", "## Читання аудіо за `location`"),
            cell("code", AUDIO),
            cell("markdown", "## Ознаки: декодування, log-mel, кореляція"),
            cell("code", FEATURES),
            cell("markdown", "## Порівняння пар"),
            cell("code", SCORE),
            cell("markdown", "## Прослуховування"),
            cell("code", UI),
            cell("markdown", "## Записати знахідки в `acoustic_matches.csv` (для препроцесингу)"),
            cell("code", EXPORT),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"cell{index:02d}"
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
