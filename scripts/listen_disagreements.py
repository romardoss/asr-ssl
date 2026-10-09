"""Play clips whose MMS language disagrees with the Whisper label.

Reads dataset/language_logits.csv and dataset/voa_ukr_meta.json. Prints both
labels, writes the MP3 from the local Hugging Face snapshot, and opens it in
the default player. Enter plays the next clip. q stops.

    python listen_disagreements.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGITS_PATH = ROOT / "dataset" / "language_logits.csv"
META_PATH = ROOT / "dataset" / "voa_ukr_meta.json"
DATASET_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--robinhad--VOA-ukr"

# Whisper ISO 639-1 codes mapped onto the MMS ISO 639-3 columns in the logits file.
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
    "yue": "yue",
}


def whisper_as_mms(code: str, columns: set[str]) -> str:
    code = (code or "").strip().lower()
    if code in columns:
        return code
    return WHISPER_TO_MMS.get(code, code)


def load_predictions(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as handle:
        languages = handle.readline().rstrip("\r\n").split(",")[1:]
        rows = []
        for record in csv.reader(handle):
            scores = [float(value) for value in record[1:]]
            winner = max(range(len(scores)), key=scores.__getitem__)
            peak = scores[winner]
            normalizer = math.fsum(math.exp(score - peak) for score in scores)
            rows.append(
                {
                    "ID": record[0],
                    "mms_language": languages[winner],
                    "confidence": 1.0 / normalizer,
                }
            )
    return languages, rows


def load_whisper_languages(path: Path) -> dict[str, dict]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    by_id = {}
    for item in meta["files"]:
        if item.get("error") or not item.get("language") or not item.get("path"):
            continue
        clip_id = Path(item["path"]).stem
        by_id[clip_id] = {
            "path": item["path"],
            "stream_index": item.get("stream_index"),
            "whisper_language": item["language"],
        }
    return by_id


def mismatched() -> list[dict]:
    whisper = load_whisper_languages(META_PATH)
    languages, predictions = load_predictions(LOGITS_PATH)
    columns = set(languages)
    rows = []
    for prediction in predictions:
        meta = whisper.get(prediction["ID"])
        if meta is None:
            continue
        whisper_language = meta["whisper_language"]
        if whisper_as_mms(whisper_language, columns) == prediction["mms_language"]:
            continue
        rows.append(
            {
                "ID": prediction["ID"],
                "path": meta["path"],
                "stream_index": meta["stream_index"],
                "whisper_language": whisper_language,
                "mms_language": prediction["mms_language"],
                "confidence": prediction["confidence"],
            }
        )
    rows.sort(key=lambda row: (row["stream_index"] is None, row["stream_index"] or 0))
    return rows


def dataset_cache() -> tuple[Path, str]:
    cache = DATASET_CACHE
    if not cache.is_dir():
        cache = Path("/home/romardoss/.cache/huggingface/hub/datasets--robinhad--VOA-ukr")
    if not cache.is_dir():
        raise SystemExit(f"dataset cache not found: {DATASET_CACHE}")
    revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    return cache, revision


def shards() -> list[Path]:
    cache, revision = dataset_cache()
    snapshot = cache / "snapshots" / revision / "data"
    tree = json.loads((cache / "trees" / f"{revision}.json").read_text(encoding="utf-8"))
    files = []
    missing = []
    for name in sorted(tree["files"]):
        if not (name.startswith("data/train-") and name.endswith(".parquet")):
            continue
        link = snapshot / Path(name).name
        blob = cache / "blobs" / tree["files"][name]["blob_id"]
        chosen = None
        for candidate in (link, blob):
            try:
                if candidate.stat().st_size > 0:
                    chosen = candidate
                    break
            except OSError:
                continue
        if chosen is None:
            missing.append(name)
        else:
            files.append(chosen)
    if not files:
        sample = missing[0] if missing else "data/train-*.parquet"
        raise SystemExit(f"parquet shards are not downloaded under {snapshot} (first missing: {sample})")
    if missing:
        print(f"{len(missing)} shards are listed but not downloaded")
    return files


def _leaf_column(table, leaf: str):
    if leaf in table.column_names:
        return table.column(leaf)
    dotted = f"audio.{leaf}"
    if dotted in table.column_names:
        column = table.column(dotted)
        if hasattr(column, "field"):
            try:
                return column.field(leaf)
            except (KeyError, AttributeError):
                return column
        return column
    audio = table.column("audio")
    return audio.field(leaf)


def _read_leaf(shard: Path, leaf: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read the dataset shards") from exc

    dotted = f"audio.{leaf}"
    try:
        table = pq.read_table(shard, columns=[dotted])
    except Exception:
        table = pq.read_table(shard, columns=["audio"])
    return _leaf_column(table, leaf)


def iter_shard_clips(shard: Path, wanted: set[str]):
    names = _read_leaf(shard, "path").to_pylist()
    hits = [index for index, name in enumerate(names) if Path(name or "").stem in wanted]
    if not hits:
        return
    blobs = _read_leaf(shard, "bytes")
    for index in hits:
        raw = blobs[index].as_py()
        yield names[index] or "", raw if isinstance(raw, (bytes, bytearray)) else b""


def open_audio(path: Path) -> None:
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if os.environ.get("WSL_DISTRO_NAME") and shutil.which("wslpath") and shutil.which("powershell.exe"):
        windows_path = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", "Start-Process", "-FilePath", windows_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    opener = shutil.which("xdg-open") or shutil.which("gio") or shutil.which("ffplay")
    if opener is None:
        print(f"no player found; file is {path}")
        return
    name = Path(opener).name
    if name == "gio":
        command = [opener, "open", str(path)]
    elif name == "ffplay":
        command = [opener, "-autoexit", str(path)]
    else:
        command = [opener, str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def listen(rows: list[dict]) -> None:
    if not rows:
        print("no clips with different language labels")
        return
    wanted = {row["ID"]: row for row in rows}
    folder = Path(tempfile.mkdtemp(prefix="voa-listen-"))
    print(f"{len(wanted)} clips, files in {folder}")
    found = 0
    for shard in shards():
        if not wanted:
            break
        for name, raw in iter_shard_clips(shard, set(wanted)):
            clip_id = Path(name).stem
            if clip_id not in wanted:
                continue
            row = wanted.pop(clip_id)
            if not raw:
                print(f"missing audio bytes for {name}")
                continue
            found += 1
            suffix = Path(name).suffix or ".mp3"
            clip_path = folder / f"{clip_id}{suffix}"
            clip_path.write_bytes(raw)
            print(
                f"\n[{found}/{found + len(wanted)}] {name}\n"
                f"whisper: {row['whisper_language']}\n"
                f"mms:     {row['mms_language']}\n"
                f"confidence: {row['confidence']:.3f}"
            )
            open_audio(clip_path)
            if wanted:
                answer = input("Enter for the next clip, q to stop: ").strip().lower()
                if answer == "q":
                    return
    if wanted:
        print("not found in the dataset:", ", ".join(sorted(wanted)))


def main() -> None:
    listen(mismatched())


if __name__ == "__main__":
    main()
