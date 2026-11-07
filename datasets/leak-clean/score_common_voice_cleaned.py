"""Compare WER on the Common Voice test with WER on its cleaned part.

A closed check, not part of training or evaluation. ``train_validated.tsv``
keeps speakers and sentences it shares with ``test.tsv``. ``test_cleaned.tsv``
(written by ``clean_common_voice_train.py``) is the test rows whose speaker and
normalized sentence are both absent from train. If WER on that part is close
to WER on the full test, the shared speakers and sentences do not inflate the
result.

Nothing is transcribed again: the cleaned clips are a subset of the test, so
their hypotheses come from the ``common_voice`` rows of each run's
``predictions.csv``. Text is normalized as in ``fine-tune/fine-tune.py``.

    python datasets/leak-clean/score_common_voice_cleaned.py
    python datasets/leak-clean/score_common_voice_cleaned.py evaluation/baseline
    python datasets/leak-clean/score_common_voice_cleaned.py --csv cleaned_scores.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jiwer
import pandas as pd

from clean_common_voice_train import CLEANED_TEST_NAME, DEFAULT_SPLITS, ROOT, normalize_text, read_split

DATASET = "common_voice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        default=[ROOT / "evaluation"],
        help="Evaluation runs, or folders searched for runs (default: evaluation/).",
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="Folder with test_cleaned.tsv.")
    parser.add_argument("--csv", type=Path, help="Also write the table to this CSV file.")
    return parser.parse_args()


def rates(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    # Same rule as corpus_metrics in fine-tune/fine-tune.py: empty references are skipped.
    pairs = [(normalize_text(ref), normalize_text(hyp)) for ref, hyp in zip(references, hypotheses)]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref]
    refs, hyps = [ref for ref, _ in pairs], [hyp for _, hyp in pairs]
    return jiwer.wer(refs, hyps) * 100, jiwer.cer(refs, hyps) * 100


def run_dirs(folders: list[Path]) -> list[Path]:
    found = set()
    for folder in folders:
        folder = folder if folder.is_absolute() else Path.cwd() / folder
        if (folder / "predictions.csv").is_file():
            found.add(folder.resolve())
        else:
            found.update(path.parent.resolve() for path in folder.rglob("predictions.csv"))
    if not found:
        raise SystemExit(f"No predictions.csv under {', '.join(str(folder) for folder in folders)}")
    return sorted(found)


def main() -> None:
    args = parse_args()
    splits = args.splits if args.splits.is_absolute() else ROOT / args.splits
    cleaned = set(read_split(splits / CLEANED_TEST_NAME)["path"])

    rows = []
    for run in run_dirs(args.folders):
        predictions = pd.read_csv(run / "predictions.csv", dtype=str, keep_default_na=False)
        voice = predictions[predictions["dataset"] == DATASET]
        if voice.empty:
            print(f"skip {run}: no {DATASET} rows")
            continue
        chosen = voice[voice["audio_path"].map(lambda path: Path(path).name).isin(cleaned)]
        if len(chosen) != len(cleaned):
            raise SystemExit(f"{run}: found {len(chosen)} of {len(cleaned)} cleaned clips in predictions.csv")
        full_wer, full_cer = rates(voice["reference"].tolist(), voice["hypothesis"].tolist())
        clean_wer, clean_cer = rates(chosen["reference"].tolist(), chosen["hypothesis"].tolist())
        name = run.relative_to(ROOT) if run.is_relative_to(ROOT) else run
        rows.append(
            {
                "run": str(name),
                "test_clips": len(voice),
                "cleaned_clips": len(chosen),
                "test_wer": round(full_wer, 2),
                "cleaned_wer": round(clean_wer, 2),
                "wer_diff": round(clean_wer - full_wer, 2),
                "test_cer": round(full_cer, 2),
                "cleaned_cer": round(clean_cer, 2),
                "cer_diff": round(clean_cer - full_cer, 2),
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.csv is not None:
        table.to_csv(args.csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
