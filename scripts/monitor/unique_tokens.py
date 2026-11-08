"""Find characters, words, and Whisper tokens of one dataset that the others never use.

Only transcripts are read; audio is not decoded. The target dataset
(``ukr_dialects`` by default) is compared with every other labelled dataset
from ``fine-tune/dataset_roles.py``. Three levels are checked:

1. characters of the raw transcript (stress marks, brackets, Latin letters, ...);
2. words after the evaluation normalization (lowercase, punctuation removed);
3. Whisper BPE token ids of the normalized text (``--whisper``).

For each level the script prints how many distinct items are unique to the
target, what share of the target's running items they make up, and the most
frequent ones with an example sentence. Full lists go to ``--out`` as CSV.

    python scripts/monitor/unique_tokens.py
    python scripts/monitor/unique_tokens.py --target-splits test --reference-splits train --whisper
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINE_TUNE = REPO / "fine-tune"
if str(FINE_TUNE) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE))

from dataset_roles import ROLES  # noqa: E402

# Same normalization as normalize_text in fine-tune/fine-tune.py, which scores WER.
_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")
ALL_SPLITS = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="ukr_dialects", help="dataset to inspect")
    parser.add_argument("--target-splits", nargs="+", default=list(ALL_SPLITS), choices=ALL_SPLITS)
    parser.add_argument(
        "--reference",
        nargs="+",
        default=None,
        help="datasets to compare with (default: every other dataset)",
    )
    parser.add_argument("--reference-splits", nargs="+", default=list(ALL_SPLITS), choices=ALL_SPLITS)
    parser.add_argument("--whisper", action="store_true", help="also compare Whisper token ids (model.id)")
    parser.add_argument("--top", type=int, default=40, help="rows to print per level")
    parser.add_argument("--out", type=Path, default=REPO / "analysis" / "unique-tokens")
    return parser.parse_args()


def part_texts(part) -> list[str]:
    """Transcripts of one loaded dataset part, without touching the audio."""
    if hasattr(part, "sentences"):
        return [str(text) for text in part.sentences]
    if hasattr(part, "records"):
        return [record[2] for record in part.records]
    return [row[3] for row in part.rows]


def load_texts(settings, name: str, splits: list[str]) -> dict[str, list[str]]:
    from speech_data import _part

    texts = {}
    for role in splits:
        if role not in ROLES[name]:
            continue
        texts[role] = part_texts(_part(settings, name, role))
        print(f"  {name}/{role}: {len(texts[role])} transcripts", flush=True)
    return texts


def character_name(char: str) -> str:
    return unicodedata.name(char, f"U+{ord(char):04X}")


def compare(
    level: str,
    target_items: list[tuple[str, list]],
    reference: Counter,
    show,
    top: int,
    out: Path,
) -> None:
    counts: Counter = Counter()
    examples: dict = {}
    for sentence, items in target_items:
        for item in items:
            counts[item] += 1
            examples.setdefault(item, sentence)
    unique = {item: count for item, count in counts.items() if item not in reference}
    running = sum(counts.values())
    unique_running = sum(unique.values())
    sentences = sum(any(item in unique for item in items) for _sentence, items in target_items)
    print(f"\n=== {level} ===")
    print(f"distinct in target: {len(counts)}, absent from reference: {len(unique)} ({len(unique) / max(len(counts), 1):.1%})")
    print(f"running {level}: {running}, of them unseen in reference: {unique_running} ({unique_running / max(running, 1):.2%})")
    print(f"transcripts with at least one unseen {level[:-1]}: {sentences} of {len(target_items)} ({sentences / max(len(target_items), 1):.1%})")
    ranked = sorted(unique.items(), key=lambda pair: (-pair[1], str(pair[0])))
    for item, count in ranked[:top]:
        print(f"  {count:6d}  {show(item)}  | {examples[item][:90]}")
    path = out / f"{level}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "count", "example"])
        for item, count in ranked:
            writer.writerow([show(item), count, examples[item]])
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    if args.target not in ROLES:
        raise SystemExit(f"Unknown dataset {args.target}. Known: {', '.join(sorted(ROLES))}")
    references = args.reference or [name for name in ROLES if name != args.target]

    from settings import load_settings

    settings = load_settings()
    print("target:", flush=True)
    target = [text for texts in load_texts(settings, args.target, args.target_splits).values() for text in texts]
    print("reference:", flush=True)
    reference = []
    for name in references:
        reference += [text for texts in load_texts(settings, name, args.reference_splits).values() for text in texts]
    if not target or not reference:
        raise SystemExit("No transcripts to compare.")
    args.out.mkdir(parents=True, exist_ok=True)

    compare(
        "characters",
        [(text, list(text)) for text in target],
        Counter(char for text in reference for char in text),
        lambda char: f"{char!r} {character_name(char)}",
        args.top,
        args.out,
    )
    compare(
        "words",
        [(text, normalize_text(text).split()) for text in target],
        Counter(word for text in reference for word in normalize_text(text).split()),
        str,
        args.top,
        args.out,
    )
    if args.whisper:
        from transformers import WhisperTokenizer

        tokenizer = WhisperTokenizer.from_pretrained(settings.model_id)

        def ids(text: str) -> list[int]:
            return tokenizer.encode(" " + normalize_text(text), add_special_tokens=False)

        compare(
            "whisper_tokens",
            [(text, ids(text)) for text in target],
            Counter(token for text in reference for token in ids(text)),
            lambda token: f"{token} {tokenizer.decode([token])!r}",
            args.top,
            args.out,
        )


if __name__ == "__main__":
    main()
