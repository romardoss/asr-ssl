"""Generate ``review_leaks.ipynb``: listen to the clips the leak scripts flag.

The notebook does not reimplement the detection. It imports ``repeat_groups``
from ``find_overlaps.py`` and ``load_compared`` / ``match`` from
``find_audio_matches.py`` and runs them on ``metadata.parquet`` in memory, so the
groups are the same as in ``duplicates.csv`` and ``audio_matches.csv``. Nothing
is written except the verdicts you save yourself.

Run it on the machine that has the audio (the paths in ``location`` are paths
on that machine), from any folder:

    python datasets/leak-clean/make_review_notebook.py
    jupyter lab datasets/leak-clean/review_leaks.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().with_name("review_leaks.ipynb")

INTRO = """\
# Ручне прослуховування клипів, позначених як витік

Ноутбук бере `metadata.parquet` і ті самі функції, що й скрипти (`find_overlaps.repeat_groups`,
`find_audio_matches.load_compared` / `match`), тож групи збігаються з `duplicates.csv` і `audio_matches.csv`.
Витоком вважається dev/test-клип, який має пару в train: за SHA-256 аудіо, за іменем файлу або за текстом.

Для кожного такого клипу показано сам клип і до `PARTNERS` його train-пар з плеєрами. Аудіо читається
з диска сервера (`location`) і вбудовується в сторінку, тому ноутбук треба запускати на машині з даними.
Плеєр грає в браузері, у тому числі через SSH-тунель чи VS Code Remote-SSH.

Перед збереженням чи комітом очистіть виводи (`Edit → Clear Outputs`): вони містять аудіо в base64.
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
import ipywidgets as widgets
from IPython.display import HTML, display

# Folder with the leak-clean scripts: found from the notebook's folder upwards.
LEAK_DIR = next(
    folder / name
    for folder in [Path.cwd(), *Path.cwd().parents]
    for name in ("", "datasets/leak-clean")
    if (folder / name / "leak_common.py").is_file()
)
sys.path.insert(0, str(LEAK_DIR))

from find_audio_matches import load_compared, match
from find_overlaps import repeat_groups
from leak_common import output_dir

FOLDER = Path(os.environ["LEAK_OUTPUT"]) if os.environ.get("LEAK_OUTPUT") else output_dir(None)
print("scripts:", LEAK_DIR)
print("metadata:", FOLDER)
'''

PARAMS = '''\
# What to review. Re-run this cell and everything below after changing it.
MODE = "sha256"        # "sha256": same audio bytes | "name": same file name | "text": same normalized text
NAME_SAME_BYTES = None # only for MODE = "name": None (all), or a list such as ["no", "unknown"]
MIN_WORDS = 4          # only for MODE = "text": as --min-words in find_overlaps.py
UNITS = None           # None (all dev/test splits) or a list such as ["common_voice/test", "cv10_test/test"]
PER_UNIT = 20          # random sample of flagged clips per dev/test split; None keeps all
PARTNERS = 2           # train partners shown for each flagged clip
SEED = 0
PAGE_SIZE = 5          # pairs per page
'''

CANDIDATES = '''\
metadata = load_compared(FOLDER)  # train/dev/test only, with ``unit``, as in find_audio_matches.py

if MODE in ("sha256", "name"):
    matches, ignored = match(metadata)  # the same call that writes audio_matches.csv
    frame = matches[matches["key"] == MODE].copy()
    if MODE == "name" and NAME_SAME_BYTES is not None:
        frame = frame[frame["same_bytes"].isin(NAME_SAME_BYTES)]
elif MODE == "text":
    compared = metadata[metadata["words"] >= MIN_WORDS]  # as in find_overlaps.find
    frame, _ = repeat_groups(compared, "text_norm")
    frame["key"] = "text"
    frame["same_bytes"] = ""
else:
    raise SystemExit(f"unknown MODE {MODE!r}")

# Groups are numbered per key, so a group is (key, group).
frame = frame.reset_index(drop=True)
has_train = frame["units"].map(lambda units: any(unit.endswith("/train") for unit in units.split("; "))).astype(bool)
frame["unit"] = frame["dataset"] + "/" + frame["split"]
flagged = frame[has_train & (frame["split"] != "train")]
if UNITS is not None:
    flagged = flagged[flagged["unit"].isin(UNITS)]

train_by_group = {key: rows for key, rows in frame[frame["split"] == "train"].groupby(["key", "group"])}

print(f"{MODE}: flagged dev/test clips per split (all of them, before sampling)")
print(flagged.groupby("unit").size().to_string() if len(flagged) else "nothing flagged")

if PER_UNIT is not None:
    flagged = flagged.sample(frac=1, random_state=SEED).groupby("unit").head(PER_UNIT)
flagged = flagged.sort_values(["unit", "group", "id"]).reset_index(drop=True)
print(f"to review: {len(flagged)} clips")
'''

AUDIO = '''\
def _audio_bytes(column, row):
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column[row].as_py()["bytes"]


def load_audio(dataset: str, location: str) -> bytes:
    """Encoded audio bytes of a clip, from the ``location`` written by collect_metadata.py."""
    if dataset == "voa":
        raise NotImplementedError("VOA slices are cut from long files and have no clip of their own")
    if "#batch=" in location:  # FLEURS: Arrow stream file, batch and row
        path, _, query = location.partition("#")
        params = dict(item.split("=") for item in query.split("&"))
        for index, batch in enumerate(ipc.open_stream(path)):
            if index == int(params["batch"]):
                return _audio_bytes(batch.column("audio"), int(params["row"]))
        raise LookupError(location)
    if "#rowgroup=" in location:  # parquet: row group and row
        path, _, query = location.partition("#")
        params = dict(item.split("=") for item in query.split("&"))
        table = pq.ParquetFile(path).read_row_group(int(params["rowgroup"]), columns=["audio"])
        return _audio_bytes(table.column("audio"), int(params["row"]))
    if ".zip!" in location:  # OpenTTS: member of a zip archive
        archive, _, member = location.partition(".zip!")
        with zipfile.ZipFile(archive + ".zip") as zipped:
            return zipped.read(member)
    return Path(location).read_bytes()  # Common Voice: a file in clips/


def mime_of(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return "audio/wav"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/mpeg"  # ID3 tag or an MPEG frame


def player(dataset: str, location: str) -> str:
    try:
        data = load_audio(dataset, location)
    except Exception as error:  # a missing file should not stop the review
        return f"<i>no audio: {html.escape(str(error))}</i>"
    encoded = base64.b64encode(data).decode()
    return f'<audio controls preload="none" src="data:{mime_of(data)};base64,{encoded}"></audio> <small>{len(data) // 1024} KB</small>'
'''

UI = '''\
verdicts: dict[tuple, str] = {}
CHOICES = ["?", "leak", "not a leak", "unsure"]


def clip_row(role: str, clip) -> str:
    text = html.escape(str(clip.get("text_norm", "")))
    return (
        "<tr>"
        f"<td><b>{role}</b></td>"
        f"<td>{html.escape(clip['unit'])}</td>"
        f"<td><small>{html.escape(str(clip['id']))}</small></td>"
        f"<td>{text}</td>"
        f"<td>{player(clip['dataset'], clip['location'])}</td>"
        "</tr>"
    )


def pair_widget(clip) -> widgets.Widget:
    partners = train_by_group.get((clip["key"], clip["group"]))
    rows = [clip_row("dev/test", clip)]
    if partners is not None:
        partners = partners.sample(min(len(partners), PARTNERS), random_state=SEED)
        rows += [clip_row("train", partner) for _, partner in partners.iterrows()]
    note = f"{clip['key']} group {clip['group']}, {clip['copies']} copies in {html.escape(clip['units'])}"
    if clip["same_bytes"]:
        note += f", same_bytes={clip['same_bytes']}"
    card = widgets.HTML(f"<p><b>{note}</b></p><table>{''.join(rows)}</table>")
    key = (clip["key"], clip["group"], clip["id"])
    verdict = widgets.ToggleButtons(options=CHOICES, value=verdicts.get(key, "?"), description="Verdict:")
    verdict.observe(lambda change: verdicts.__setitem__(key, change["new"]), names="value")
    return widgets.VBox([card, verdict], layout=widgets.Layout(border="1px solid #8885", padding="6px", margin="6px 0"))


view = widgets.Output()
state = {"page": 0}
pages = max(1, -(-len(flagged) // PAGE_SIZE))
label = widgets.Label()
previous_button = widgets.Button(description="< prev")
next_button = widgets.Button(description="next >")


def render() -> None:
    start = state["page"] * PAGE_SIZE
    label.value = f"page {state['page'] + 1} of {pages}, clips {start + 1}-{min(start + PAGE_SIZE, len(flagged))} of {len(flagged)}"
    view.clear_output(wait=True)
    with view:
        for _, clip in flagged.iloc[start : start + PAGE_SIZE].iterrows():
            display(pair_widget(clip))


def turn(step: int):
    def handler(_):
        state["page"] = min(max(state["page"] + step, 0), pages - 1)
        render()

    return handler


previous_button.on_click(turn(-1))
next_button.on_click(turn(1))
display(widgets.HBox([previous_button, next_button, label]), view)
render()
'''

SAVE = '''\
# Save the verdicts so far (next to the other leak-clean outputs, which git ignores).
rows = [
    {"key": key, "group": group, "id": clip_id, "verdict": verdict}
    for (key, group, clip_id), verdict in verdicts.items()
    if verdict != "?"
]
verdicts_path = FOLDER / f"review_verdicts_{MODE}.csv"
pd.DataFrame(rows, columns=["key", "group", "id", "verdict"]).to_csv(verdicts_path, index=False)
print(f"wrote {verdicts_path} ({len(rows)} verdicts)")
print(pd.Series([row["verdict"] for row in rows]).value_counts().to_string() if rows else "no verdicts yet")
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
            cell("markdown", "## Кандидати (та сама логіка, що в скриптах)"),
            cell("code", CANDIDATES),
            cell("markdown", "## Читання аудіо за `location`"),
            cell("code", AUDIO),
            cell("markdown", "## Прослуховування"),
            cell("code", UI),
            cell("markdown", "## Зберегти вердикти"),
            cell("code", SAVE),
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
