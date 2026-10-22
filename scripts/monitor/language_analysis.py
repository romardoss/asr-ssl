"""Plot MMS language-id results against the Whisper labels in the metadata.

Reads datasets/voa-ukr/mms-language-confidences.csv. The predicted language is the argmax
logit. Confidence is that language's softmax probability. Whisper labels in
datasets/voa-ukr/voa_ukr_meta.json use ISO 639-1 codes; MMS uses ISO 639-3, so the
comparison maps Whisper codes before counting disagreements.

    python scripts/monitor/language_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGITS_PATH = ROOT / "datasets" / "voa-ukr" / "mms-language-confidences.csv"
META_PATH = ROOT / "datasets" / "voa-ukr" / "voa_ukr_meta.json"
FIGURE_DIR = ROOT / "analysis" / "voa-meta-plots"

# Whisper language codes to the MMS-LID-256 codes used in the logit columns.
WHISPER_TO_MMS = {
    "en": "eng",
    "zh": "cmn",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "ko": "kor",
    "fr": "fra",
    "ja": "jpn",
    "pt": "por",
    "tr": "tur",
    "pl": "pol",
    "ca": "cat",
    "nl": "nld",
    "ar": "ara",
    "sv": "swe",
    "it": "ita",
    "id": "ind",
    "hi": "hin",
    "fi": "fin",
    "vi": "vie",
    "he": "heb",
    "uk": "ukr",
    "el": "ell",
    "ms": "zlm",
    "cs": "ces",
    "ro": "ron",
    "da": "dan",
    "hu": "hun",
    "ta": "tam",
    "no": "nob",
    "th": "tha",
    "ur": "urd",
    "hr": "hrv",
    "bg": "bul",
    "lt": "lit",
    "la": "lat",
    "mi": "mri",
    "ml": "mal",
    "cy": "cym",
    "sk": "slk",
    "te": "tel",
    "fa": "fas",
    "lv": "lav",
    "bn": "ben",
    "sr": "srp",
    "az": "aze",
    "sl": "slv",
    "kn": "kan",
    "et": "est",
    "mk": "mkd",
    "br": "bre",
    "eu": "eus",
    "is": "isl",
    "hy": "hye",
    "ne": "npi",
    "mn": "mon",
    "bs": "bos",
    "kk": "kaz",
    "sq": "sqi",
    "sw": "swh",
    "gl": "glg",
    "mr": "mar",
    "pa": "pan",
    "si": "sin",
    "km": "khm",
    "sn": "sna",
    "yo": "yor",
    "so": "som",
    "af": "afr",
    "oc": "oci",
    "ka": "kat",
    "be": "bel",
    "tg": "tgk",
    "sd": "snd",
    "gu": "guj",
    "am": "amh",
    "yi": "yid",
    "lo": "lao",
    "uz": "uzb",
    "fo": "fao",
    "ht": "hat",
    "ps": "pus",
    "tk": "tuk",
    "nn": "nno",
    "mt": "mlt",
    "sa": "san",
    "lb": "ltz",
    "my": "mya",
    "bo": "bod",
    "tl": "tgl",
    "mg": "mlg",
    "as": "asm",
    "tt": "tat",
    "haw": "haw",
    "ln": "lin",
    "ha": "hau",
    "ba": "bak",
    "jw": "jav",
    "su": "sun",
}


def save_fig(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=120, bbox_inches="tight")
    plt.close()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"logits not found: {path}")
    table = pd.read_csv(path)
    if "ID" not in table.columns:
        raise ValueError(f"{path} has no ID column")
    languages = [column for column in table.columns if column != "ID"]
    logits = table[languages].apply(pd.to_numeric, errors="coerce")
    valid = logits.notna().all(axis=1)
    logits = logits.loc[valid].to_numpy(dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    choice = probabilities.argmax(axis=1)
    return pd.DataFrame(
        {
            "ID": table.loc[valid, "ID"].astype(str).to_numpy(),
            "mms_language": np.array(languages, dtype=object)[choice],
            "confidence": probabilities.max(axis=1),
        }
    )


def load_whisper_languages(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["files"] if isinstance(payload, dict) else payload
    languages = {}
    for row in rows:
        name = row.get("path")
        if not name:
            continue
        languages[Path(str(name)).stem] = row.get("language")
    return pd.Series(languages, name="whisper_language")


def plot_language_pie(predictions: pd.DataFrame) -> None:
    counts = predictions["mms_language"].value_counts()
    shown = counts.head(5)
    other = int(counts.iloc[5:].sum())
    if other:
        shown = pd.concat([shown, pd.Series({"other": other})])
    plt.figure(figsize=(8, 6))
    wedges, _ = plt.pie(shown.values, startangle=90)
    plt.legend(
        wedges,
        [f"{name} ({count})" for name, count in shown.items()],
        title="MMS language",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.title("MMS language")
    save_fig("language_pie.png")


def plot_confidence(predictions: pd.DataFrame) -> None:
    bins = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0000001]
    labels = ["<0.5", "0.5–0.8", "0.8–0.9", "0.9–0.95", "0.95–0.99", "≥0.99"]
    groups = pd.cut(predictions["confidence"], bins=bins, labels=labels, right=False)
    counts = groups.value_counts().reindex(labels, fill_value=0)
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("clips")
    plt.xlabel("softmax probability of the predicted language")
    plt.title("MMS confidence")
    save_fig("confidence.png")


def compare_labels(predictions: pd.DataFrame, whisper: pd.Series) -> pd.DataFrame:
    """Attach Whisper labels and a differ flag. Whisper codes are mapped to MMS codes."""
    compared = predictions.copy()
    compared["whisper_language"] = compared["ID"].map(whisper)
    compared["whisper_mms"] = compared["whisper_language"].map(
        lambda code: WHISPER_TO_MMS.get(code, code) if pd.notna(code) else code
    )
    known = compared["whisper_mms"].notna()
    compared["differ"] = known & (compared["mms_language"] != compared["whisper_mms"])
    return compared


def plot_disagreement(predictions: pd.DataFrame, whisper: pd.Series) -> None:
    compared = compare_labels(predictions, whisper)
    differ = compared["differ"]
    known = compared["whisper_mms"].notna()
    agree = known & ~differ
    missing = ~known
    counts = pd.Series(
        {
            "agree": int(agree.sum()),
            "differ": int(differ.sum()),
            "no whisper label": int(missing.sum()),
        }
    )
    if counts["no whisper label"] == 0:
        counts = counts.drop(labels=["no whisper label"])
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("clips")
    plt.title("MMS vs Whisper language")
    for index, value in enumerate(counts.values):
        plt.text(index, value, str(value), ha="center", va="bottom")
    save_fig("language_agreement.png")
    print(counts.to_string())
    mismatches = compared.loc[differ, ["ID", "whisper_language", "mms_language"]]
    if not mismatches.empty:
        print("differ")
        print(
            mismatches.groupby(["whisper_language", "mms_language"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )


def main() -> None:
    predictions = load_predictions(LOGITS_PATH)
    print(LOGITS_PATH)
    print("clips", len(predictions))
    plot_language_pie(predictions)
    plot_confidence(predictions)
    plot_disagreement(predictions, load_whisper_languages(META_PATH))
    print("figures", FIGURE_DIR)


if __name__ == "__main__":
    main()
