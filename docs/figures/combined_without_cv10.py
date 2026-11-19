"""Перерахунок combined_weighted WER/CER без cv10_test для моделей з тез.

    env/bin/python docs/figures/combined_without_cv10.py

Читає evaluation/<run>/predictions.csv і рахує метрики так само, як
fine-tune/fine-tune.py (corpus_metrics): нормалізація normalize_text, порожні
еталони відкидаються, WER і CER — за всіма парами разом (jiwer). Перевіряє, що з
cv10_test виходить те саме число, що й у results.csv, і пише
docs/figures/combined_without_cv10.csv.
"""

import re
from pathlib import Path

import jiwer
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("combined_without_cv10.csv")

RUNS = [
    "baseline/original-whisper-medium",
    "baseline/common-voice-only-baseline",
    "student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01",
    "student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01",
    "student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01",
    "student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01",
    "whisper-uk-cv-fixed-epoch00-01",
    "whisper-m-fixed-leaks-5-epochs-epoch00-01",
    "whisper-m-fixed-leaks-5-epochs-epoch04-01",
    "whisper-m-fixed-leaks-10-epochs-epoch09-01",
]
KEEP = ("common_voice", "fleurs", "ukr_dialects")

# Те саме, що normalize_text у fine-tune/fine-tune.py.
_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def pooled(frame: pd.DataFrame) -> tuple[float, float]:
    refs, hyps = [], []
    for ref, hyp in zip(frame["reference"].fillna(""), frame["hypothesis"].fillna("")):
        ref = normalize_text(ref)
        if ref:
            refs.append(ref)
            hyps.append(normalize_text(hyp))
    return jiwer.wer(refs, hyps) * 100, jiwer.cer(refs, hyps) * 100


results = pd.read_csv(ROOT / "evaluation" / "results.csv").set_index("run")
rows = []
for run in RUNS:
    preds = pd.read_csv(ROOT / "evaluation" / run / "predictions.csv", dtype=str, keep_default_na=False)
    wer_all, cer_all = pooled(preds)
    expected = results.loc[run, ["wer_combined_weighted", "cer_combined_weighted"]].tolist()
    if abs(wer_all - expected[0]) > 0.01 or abs(cer_all - expected[1]) > 0.01:
        raise SystemExit(f"{run}: перерахунок з cv10 ({wer_all:.4f}, {cer_all:.4f}) не збігся з results.csv {expected}")
    wer, cer = pooled(preds[preds["dataset"].isin(KEEP)])
    rows.append({"run": run, "wer_combined_no_cv10": round(wer, 4), "cer_combined_no_cv10": round(cer, 4),
                 "clips": int(preds["dataset"].isin(KEEP).sum())})
    print(f"{run}: WER {wer:.2f}  CER {cer:.2f}")

pd.DataFrame(rows).to_csv(OUT, index=False)
print(OUT)
