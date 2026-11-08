"""Score Ukrainian pseudo-labels with an OpenRouter judge.

Two backends share ``prompt.txt`` and the 0-5 marks:

1. ``llm`` (default): a chat model, ``OPENROUTER_MODEL``. Batches of
   transcripts go out as ``index label`` lines and come back as ``index mark``.
2. ``decisions``: a decision model, ``OPENROUTER_DECISION_MODEL``, through the
   Decisions API (``decisions.py``). One request per transcript; the mark is
   the most probable level, and ``score`` / ``confidence`` are stored too.

``USE_DECISION_INSTEAD_OF_LLM`` picks the backend; ``--backend`` overrides it.
``OPENROUTER_API_KEY`` is used by both. ``.env`` fills any name that is not set.

Several requests stay in flight, and each accepted batch is appended to
``label_marks.csv`` beside the source file. A stopped run continues from
indexes that are not in that file yet. Without ``--labels`` the source is the
newest ``labels.csv`` of ``pseudo_labelling.teacher_model``. ``--workers`` and
``--retries`` default to ``pipeline.judge_workers`` and ``pipeline.judge_retries``.

    python pseudo-labelling/llm-as-a-judge/judge.py --max-examples 32
    python pseudo-labelling/llm-as-a-judge/judge.py --backend decisions --max-examples 32
    python pseudo-labelling/llm-as-a-judge/judge.py --backend decisions --test

``--test`` checks a judge before a full run. It marks 100 random rows
(fixed seed), or 200 when some mark from 0 to 5 never appeared in the first
100, and writes ``judge_test-<model>-<time>.csv`` beside the labels for a manual
review: mark (and score/confidence for decisions), text, ID. ``label_marks.csv``
is not touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import decisions  # noqa: E402
from project_config import config_int, configured_labelling_file, load_config  # noqa: E402

CONFIG = load_config()
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
MARKS_NAME = "label_marks.csv"
BATCH_SIZE = 16
TEST_ROWS = 100
TEST_ROWS_MAX = 200
TEST_SEED = 0
MARK_FIELDS = ["index", "ID", "mark", "score", "confidence"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FORMAT_RULES = """
Reply with marks only.
The user message has one transcript per line: `<index> <transcript>`.
Return one line per transcript and no other text.
Each line must be `<index> <mark>`, with the same index and an integer mark from 0 to 5.
Use every given index once.
""".strip()

load_dotenv = decisions.load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Pseudo-label CSV to score. Default: newest labels.csv of pseudo_labelling.teacher_model.",
    )
    parser.add_argument("--max-examples", type=int, help="Score only the first N rows of the CSV.")
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Mark {TEST_ROWS} random rows ({TEST_ROWS_MAX} if a mark 0-5 is missing) into a review CSV; "
        "label_marks.csv is not touched.",
    )
    parser.add_argument(
        "--backend",
        choices=decisions.BACKENDS,
        default=None,
        help="llm or decisions. Default: decisions when USE_DECISION_INSTEAD_OF_LLM is true, else llm.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config_int(CONFIG, "pipeline.judge_workers"),
        help="How many OpenRouter requests may be in flight.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=config_int(CONFIG, "pipeline.judge_retries"),
        help="Attempts per batch before leaving it for the next run.",
    )
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


def read_labels(path: Path, limit: int | None) -> list[tuple[int, str, str]]:
    if not path.is_file():
        raise SystemExit(f"Labels file not found: {path}")
    rows: list[tuple[int, str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            rows.append((index, row["ID"], row["text"]))
    return rows


def valid_mark_rows(path: Path) -> list[dict]:
    """Rows of a marks file whose index and mark (0-5) parse.

    A run killed mid-write can leave a block of NUL bytes or a cut line; those
    rows are skipped (and reported), so their indexes count as not marked and a
    resumed run marks them again.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows, broken = [], 0
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        lines = (line.replace("\0", "") for line in handle)
        for row in csv.DictReader(lines):
            index, mark = (row.get("index") or "").strip(), (row.get("mark") or "").strip()
            if index.isdigit() and mark.isdigit() and int(mark) <= 5 and row.get("ID"):
                rows.append(row)
            else:
                broken += 1
    if broken:
        print(f"{path}: skipped {broken} damaged rows; they count as not marked", file=sys.stderr, flush=True)
    return rows


def finished_indexes(path: Path) -> set[int]:
    return {int(row["index"]) for row in valid_mark_rows(path)}


def append_marks(path: Path, rows: list[dict]) -> None:
    """Append ``{"index", "ID", "mark", ["score", "confidence"]}`` rows.

    A new file gets every column. An existing file keeps its own header, so an
    older ``index,ID,mark`` file stays readable when another backend continues it.
    """
    new_file = not path.is_file() or path.stat().st_size == 0
    fields = MARK_FIELDS
    if not new_file:
        with path.open(encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle), MARK_FIELDS)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def batch_message(batch: list[tuple[int, str, str]]) -> str:
    lines = []
    for index, _label_id, text in batch:
        flat = " ".join(text.split())
        lines.append(f"{index} {flat}")
    return "\n".join(lines)


def strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_marks(text: str, expected: set[int]) -> dict[int, int] | None:
    found: dict[int, int] = {}
    for line in strip_fence(text).splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        index = int(parts[0])
        mark = int(parts[1])
        if index not in expected or index in found or mark > 5:
            return None
        found[index] = mark
    if set(found) != expected:
        return None
    return found


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def call_openrouter(api_key: str, model: str, prompt: str, user_text: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": f"{prompt}\n\n{FORMAT_RULES}"},
                {"role": "user", "content": user_text},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    return message_text(payload["choices"][0]["message"])


class Judge:
    """One configured judge: backend, model, key, and the prompt it scores with."""

    def __init__(self, backend: str | None = None, model: str | None = None) -> None:
        load_dotenv()
        self.backend = backend or decisions.configured_backend()
        if self.backend not in decisions.BACKENDS:
            raise SystemExit(f"Unknown judge backend {self.backend!r}")
        self.model = (model or decisions.model_for(self.backend)).strip()
        self.api_key = os.environ.get(decisions.KEY_ENV, "").strip()
        if not self.api_key:
            raise SystemExit(f"{decisions.KEY_ENV} is empty.")
        if not self.model:
            name = decisions.MODEL_ENV if self.backend == "decisions" else decisions.LLM_MODEL_ENV
            raise SystemExit(f"{name} is empty.")
        if not PROMPT_PATH.is_file():
            raise SystemExit(f"Prompt not found: {PROMPT_PATH}")
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        self.instructions, self.levels = decisions.rubric(self.prompt)
        # A decision model takes one state per request.
        self.batch_size = 1 if self.backend == "decisions" else BATCH_SIZE

    def judge_batch(self, batch: list[tuple[int, str, str]], attempt: int) -> list[dict] | None:
        """Mark rows for append_marks, or None when the batch should be retried."""
        if attempt:
            time.sleep(min(8, 2 ** (attempt - 1)))
        expected = {index for index, _label_id, _text in batch}
        by_index = {index: label_id for index, label_id, _text in batch}
        try:
            if self.backend == "llm":
                reply = call_openrouter(self.api_key, self.model, self.prompt, batch_message(batch))
                marks = parse_marks(reply, expected)
                if marks is None:
                    return None
                return [{"index": i, "ID": by_index[i], "mark": mark} for i, mark in sorted(marks.items())]
            rows = []
            for index, label_id, text in batch:
                decision = decisions.decide(self.api_key, self.model, self.instructions, self.levels, " ".join(text.split()))
                if decision is None:
                    print(f"decisions: no score answer for index {index}", file=sys.stderr, flush=True)
                    return None
                rows.append(
                    {
                        "index": index,
                        "ID": label_id,
                        "mark": decision.mark,
                        "score": f"{decision.score:.4f}",
                        "confidence": f"{decision.confidence:.4f}",
                    }
                )
            return rows
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            print(f"openrouter HTTP {error.code} for indexes {sorted(expected)[:3]}...: {detail}", file=sys.stderr, flush=True)
            return None
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            print(f"openrouter request failed for indexes {sorted(expected)[:3]}...: {error}", file=sys.stderr, flush=True)
            return None

    def mark_rows(self, rows: list[tuple[int, str, str]], marks_path: Path, workers: int, retries: int) -> list[int]:
        """Mark ``(index, ID, text)`` rows not yet in ``marks_path``; return the indexes left unmarked."""
        done = finished_indexes(marks_path)
        pending = [row for row in rows if row[0] not in done]
        print(
            f"backend {self.backend} model {self.model} rows {len(rows)} "
            f"already_marked {len(rows) - len(pending)} pending {len(pending)} -> {marks_path}",
            flush=True,
        )
        if not pending:
            return []

        from tqdm import tqdm

        size = self.batch_size
        batches = [pending[start : start + size] for start in range(0, len(pending), size)]
        attempts: dict[tuple[int, ...], int] = {}
        failed: list[int] = []
        saved = 0
        progress = tqdm(total=len(pending), desc="marks", unit="label", file=sys.stderr)

        def key_of(batch):
            return tuple(index for index, _label_id, _text in batch)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            inflight = {}

            def submit_more() -> None:
                while batches and len(inflight) < workers:
                    batch = batches.pop(0)
                    inflight[executor.submit(self.judge_batch, batch, attempts.get(key_of(batch), 0))] = batch

            submit_more()
            while inflight:
                finished, _pending_futures = wait(inflight, return_when=FIRST_COMPLETED)
                for future in finished:
                    batch = inflight.pop(future)
                    key = key_of(batch)
                    marked = future.result()
                    if marked is None:
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] < retries:
                            batches.append(batch)
                        else:
                            failed.extend(key)
                            progress.update(len(batch))
                    else:
                        append_marks(marks_path, marked)
                        saved += len(marked)
                        progress.update(len(marked))
                submit_more()
        progress.close()
        print(f"saved {saved} marks -> {marks_path}", flush=True)
        if failed:
            preview = ", ".join(str(index) for index in failed[:8])
            print(f"left {len(failed)} indexes for the next run, first: {preview}", file=sys.stderr, flush=True)
        return failed


def read_marks(path: Path) -> list[dict]:
    return valid_mark_rows(path)


def run_test(judge: Judge, rows: list[tuple[int, str, str]], out_dir: Path, workers: int, retries: int) -> Path:
    """Mark a random test sample and write a review CSV sorted by mark."""
    order = list(rows)
    random.Random(TEST_SEED).shuffle(order)
    texts = {index: text for index, _label_id, text in order}
    with tempfile.TemporaryDirectory() as tmp:
        marks_path = Path(tmp) / "marks.csv"
        judge.mark_rows(order[:TEST_ROWS], marks_path, workers, retries)
        missing = sorted(set(range(6)) - {int(row["mark"]) for row in read_marks(marks_path)})
        if missing and len(order) > TEST_ROWS:
            print(f"marks {missing} did not appear in {TEST_ROWS} rows; marking up to {TEST_ROWS_MAX}", flush=True)
            judge.mark_rows(order[:TEST_ROWS_MAX], marks_path, workers, retries)
        marked = read_marks(marks_path)
    marked.sort(key=lambda row: (int(row["mark"]), int(row["index"])))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", judge.model).strip("_")
    path = out_dir / f"judge_test-{slug}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    fields = ["mark", "score", "confidence", "text", "ID", "index"]
    if judge.backend == "llm":
        fields = ["mark", "text", "ID", "index"]
    # utf-8-sig: Excel and LibreOffice open the Ukrainian text without an encoding prompt.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in marked:
            writer.writerow({**row, "text": texts[int(row["index"])]})
    counts = Counter(int(row["mark"]) for row in marked)
    print("marks: " + ", ".join(f"{mark}: {counts[mark]}" for mark in range(6)), flush=True)
    absent = [mark for mark in range(6) if not counts[mark]]
    if absent:
        print(f"never given: {absent}", flush=True)
    print(f"wrote {len(marked)} rows for review -> {path}", flush=True)
    return path


def main() -> None:
    args = parse_args()
    judge = Judge(args.backend)
    if args.labels is None:
        labels_path = configured_labelling_file("labels.csv", CONFIG)
    else:
        labels_path = args.labels if args.labels.is_absolute() else ROOT / args.labels
    if args.test:
        run_test(judge, read_labels(labels_path, args.max_examples), labels_path.parent, args.workers, args.retries)
        return
    marks_path = labels_path.parent / MARKS_NAME
    judge.mark_rows(read_labels(labels_path, args.max_examples), marks_path, args.workers, args.retries)


if __name__ == "__main__":
    main()
