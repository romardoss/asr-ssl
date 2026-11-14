"""Generate ``review_sha.ipynb``: find eval clips whose audio bytes are also in train.

Same SHA-256 of the encoded audio means the same file, so every such pair is a
leak. The notebook lists the pairs (an eval clip and a train clip with one hash)
so you can check them by ear, and writes ``sha_matches.csv`` for the preprocessing
pipeline. It cleans nothing. Its counterpart for re-encoded copies is
``review_acoustic.ipynb``.

Run it on the machine that has the audio, from any folder:

    python datasets/leak-clean/make_sha_notebook.py
    jupyter lab datasets/leak-clean/review_sha.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

from make_review_notebook import AUDIO  # load_audio / mime_of / player, shared with the other notebooks

NOTEBOOK = Path(__file__).resolve().with_name("review_sha.ipynb")

INTRO = """\
# Витік за SHA-256: той самий файл у train і в dev/test

Збіг SHA-256 закодованого аудіо означає той самий файл, тобто справжній витік (ручне прослуховування це підтвердило).
Ноутбук знаходить пари «eval-клип і train-клип з однаковим хешем», показує їх із плеєрами для перевірки
й пише `sha_matches.csv` для препроцесингу. Нічого не чистить.

Для перекодованих копій (байти різні, звук той самий) є окремий ноутбук `review_acoustic.ipynb`.

Запускати треба на машині з аудіо. Перед збереженням чи комітом очистіть виводи (в них аудіо в base64).
"""

SETUP = '''\
import base64
import html
import os
import sys
import zipfile
from pathlib import Path

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
'''

PARAMS = '''\
# What to compare. Re-run this cell and everything below after changing it.
EVAL_DATASETS = None    # None (all datasets) or a list such as ["cv10_test", "common_voice"]
EVAL_SPLITS = ["dev", "test"]
TRAIN_DATASETS = None   # None (all datasets) or a list such as ["common_voice"]
TRAIN_SPLITS = ["train"]
PAGE = 0
PAGE_SIZE = 5           # pairs per page
'''

CANDIDATES = '''\
metadata = load_compared(FOLDER)
metadata = metadata[metadata["audio_sha256"] != ""]  # VOA, common_voice/unlabelled and --no-audio-hash runs have no hash
COLUMNS = ["dataset", "split", "id", "audio_name", "audio_sha256", "location", "text_norm"]


def side(datasets, splits, suffix):
    clips = metadata[metadata["split"].isin(splits)]
    if datasets is not None:
        clips = clips[clips["dataset"].isin(datasets)]
    return clips[COLUMNS].add_suffix("_" + suffix)


eval_clips = side(EVAL_DATASETS, EVAL_SPLITS, "eval")
train_clips = side(TRAIN_DATASETS, TRAIN_SPLITS, "train")
print(f"eval clips: {len(eval_clips)}, train clips: {len(train_clips)}")

pairs = eval_clips.merge(train_clips, left_on="audio_sha256_eval", right_on="audio_sha256_train")
pairs["same_text"] = pairs["text_norm_eval"] == pairs["text_norm_train"]
pairs = pairs.sort_values(["dataset_eval", "split_eval", "dataset_train", "id_eval", "id_train"]).reset_index(drop=True)

print(f"pairs with identical SHA-256: {len(pairs)} ({pairs['id_eval'].nunique()} distinct eval clips)")
print(pairs.groupby(["dataset_eval", "split_eval", "dataset_train", "split_train"]).size().to_string() if len(pairs) else "nothing found")
'''

UI = '''\
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
    note = f"#{index}: SHA-256 {pair['audio_sha256_eval'][:12]}..., same text: {pair['same_text']}"
    return (
        '<div style="border:1px solid #8885;padding:6px;margin:6px 0">'
        f"<p>{note}</p>"
        f"<table>{clip_row('eval', pair, 'eval')}{clip_row('train', pair, 'train')}</table></div>"
    )


pages = max(1, -(-len(pairs) // PAGE_SIZE))
start = PAGE * PAGE_SIZE
print(f"page {PAGE} (valid: 0-{pages - 1}), pairs {start}-{min(start + PAGE_SIZE, len(pairs)) - 1} of {len(pairs)}")
for index in range(start, min(start + PAGE_SIZE, len(pairs))):
    display(HTML(pair_html(index, pairs.iloc[index])))
'''

EXPORT = '''\
# Writes every eval/train pair with an identical SHA-256 for the preprocessing pipeline.
columns = ["dataset_eval", "split_eval", "id_eval", "dataset_train", "split_train", "id_train", "audio_sha256_eval", "same_text"]
matches_path = FOLDER / "sha_matches.csv"
pairs[columns].rename(columns={"audio_sha256_eval": "audio_sha256"}).to_csv(matches_path, index=False)
print(f"wrote {matches_path} ({len(pairs)} pairs)")
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
            cell("markdown", "## Пари з однаковим SHA-256"),
            cell("code", CANDIDATES),
            cell("markdown", "## Читання аудіо за `location`"),
            cell("code", AUDIO),
            cell("markdown", "## Прослуховування"),
            cell("code", UI),
            cell("markdown", "## Записати знахідки в `sha_matches.csv` (для препроцесингу)"),
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
