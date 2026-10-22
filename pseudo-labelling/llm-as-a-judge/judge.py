"""Score Ukrainian pseudo-labels with an OpenRouter model.

Reads ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` from the environment
(``.env`` fills any name that is not already set). Sends batches of transcripts
as ``index label`` lines, keeps several requests in flight, and appends each
accepted batch to ``label_marks.csv`` beside the source file. A stopped run
continues from indexes that are not in that file yet.

    python pseudo-labelling/llm-as-a-judge/judge.py --max-examples 32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "output" / "pseudo-labelling" / "model-whisper-uk-epoch00-01" / "labels.csv"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
MARKS_NAME = "label_marks.csv"
BATCH_SIZE = 16
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FORMAT_RULES = """
Reply with marks only.
The user message has one transcript per line: `<index> <transcript>`.
Return one line per transcript and no other text.
Each line must be `<index> <mark>`, with the same index and an integer mark from 0 to 5.
Use every given index once.
""".strip()


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip("\"'")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Pseudo-label CSV to score.")
    parser.add_argument("--max-examples", type=int, help="Score only the first N rows of the CSV.")
    parser.add_argument("--workers", type=int, default=4, help="How many OpenRouter requests may be in flight.")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per batch before leaving it for the next run.")
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


def finished_indexes(path: Path) -> set[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["index"]) for row in csv.DictReader(handle) if row.get("index", "").isdigit()}


def append_marks(path: Path, rows: list[tuple[int, str, int]]) -> None:
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "ID", "mark"])
        if new_file:
            writer.writeheader()
        writer.writerows({"index": index, "ID": label_id, "mark": mark} for index, label_id, mark in rows)
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


def judge_batch(
    batch: list[tuple[int, str, str]],
    api_key: str,
    model: str,
    prompt: str,
    attempt: int,
) -> dict[int, int] | None:
    if attempt:
        time.sleep(min(8, 2 ** (attempt - 1)))
    expected = {index for index, _label_id, _text in batch}
    try:
        reply = call_openrouter(api_key, model, prompt, batch_message(batch))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        print(f"openrouter HTTP {error.code} for indexes {sorted(expected)[:3]}...: {detail}", file=sys.stderr, flush=True)
        return None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        print(f"openrouter request failed for indexes {sorted(expected)[:3]}...: {error}", file=sys.stderr, flush=True)
        return None
    return parse_marks(reply, expected)


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is empty.")
    if not model:
        raise SystemExit("OPENROUTER_MODEL is empty.")
    if not PROMPT_PATH.is_file():
        raise SystemExit(f"Prompt not found: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    labels_path = args.labels if args.labels.is_absolute() else ROOT / args.labels
    marks_path = labels_path.parent / MARKS_NAME
    rows = read_labels(labels_path, args.max_examples)
    done = finished_indexes(marks_path)
    pending = [row for row in rows if row[0] not in done]
    print(
        f"rows {len(rows)} already_marked {len(rows) - len(pending)} pending {len(pending)} -> {marks_path}",
        flush=True,
    )
    if not pending:
        return

    from tqdm import tqdm

    batches = [pending[start : start + BATCH_SIZE] for start in range(0, len(pending), BATCH_SIZE)]
    attempts: dict[tuple[int, ...], int] = {}
    failed: list[int] = []
    saved = 0
    progress = tqdm(total=len(pending), desc="marks", unit="label", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inflight = {}
        while batches and len(inflight) < args.workers:
            batch = batches.pop(0)
            key = tuple(index for index, _label_id, _text in batch)
            inflight[executor.submit(judge_batch, batch, api_key, model, prompt, attempts.get(key, 0))] = batch
        while inflight:
            finished, _pending_futures = wait(inflight, return_when=FIRST_COMPLETED)
            for future in finished:
                batch = inflight.pop(future)
                key = tuple(index for index, _label_id, _text in batch)
                marks = future.result()
                if marks is None:
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] < args.retries:
                        batches.append(batch)
                    else:
                        failed.extend(key)
                        progress.update(len(batch))
                else:
                    by_id = {index: label_id for index, label_id, _text in batch}
                    append_marks(marks_path, [(index, by_id[index], mark) for index, mark in sorted(marks.items())])
                    saved += len(marks)
                    progress.update(len(marks))
                while batches and len(inflight) < args.workers:
                    nxt = batches.pop(0)
                    nxt_key = tuple(index for index, _label_id, _text in nxt)
                    inflight[executor.submit(judge_batch, nxt, api_key, model, prompt, attempts.get(nxt_key, 0))] = nxt
    progress.close()
    print(f"saved {saved} marks -> {marks_path}", flush=True)
    if failed:
        preview = ", ".join(str(index) for index in failed[:8])
        print(f"left {len(failed)} indexes for the next run, first: {preview}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
