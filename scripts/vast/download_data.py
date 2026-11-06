"""Завантаження датасетів і артефактів з Hugging Face Hub на інстанс vast.ai.

Дані завантажуються на локальний диск, а не стрімляться: код читає їх з диска
з довільним доступом (row group у parquet, член zip-архіву, батч arrow), а
навчання проходить VOA кілька епох. Кожен набір кладеться в той шлях, який для
нього вказано в конфігу (``ASR_CONFIG``, на vast.ai це ``config.vast.yaml``).

Звідки що береться (секція ``downloads`` конфігу):

1. Публічні репозиторії HF уже мають той формат, який читає код, тому
   завантажуються як є, лише потрібні файли:

   1. ``opentts``: ``voices/*/train.jsonl`` і ``data/*.zip`` -> ``paths.opentts``;
   2. ``tg_voices``, ``ukr_dialects``, ``cv10_test``: ``data/*.parquet`` ->
      відповідний ``paths.*``;
   3. ``voa``: ``data/train-*.parquet`` (близько 128 ГБ) ->
      ``pseudo_labelling.voa``. Репозиторій gated: власник ``HF_TOKEN`` має
      спершу отримати доступ на сторінці датасету.

2. Приватний репозиторій ``downloads.private_repo`` містить те, чого немає
   публічно. Як його наповнити, описано в ``HF_UPLOAD.md``:

   1. ``common-voice/*.tar`` -> розпаковується в ``paths.common_voice``
      (архів містить теку ``clips/``). Після розпакування архів видаляється,
      якщо не задано ``--keep-archives``;
   2. ``fleurs-uk/`` (формат ``save_to_disk``) -> ``paths.fleurs``;
   3. ``checkpoints/`` -> ``paths.checkpoints`` (чекпоінт учителя);
   4. ``pseudo-labelling/`` -> ``pseudo_labelling.output_dir`` (``labels.csv``
      і ``label_marks.csv`` учителя).

   Файли з приватного репозиторію спершу потрапляють у тимчасову теку
   ``<paths.checkpoints>/../.hf-staging``, а потім переносяться на місце.
   Наявні файли не перезаписуються.

Скрипт можна перезапускати: набір, який уже на місці, пропускається, а
обірване завантаження продовжується. ``HF_TOKEN`` береться з оточення або з
``.env`` у корені репозиторію.

    export ASR_CONFIG=config.vast.yaml
    python scripts/vast/download_data.py                       # усі набори
    python scripts/vast/download_data.py --only voa teacher labels common_voice
    python scripts/vast/download_data.py --skip voa

Набори: common_voice, fleurs, opentts, tg_voices, ukr_dialects, cv10_test,
voa, teacher, labels.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (  # noqa: E402
    CONFIG_PATH,
    config_file,
    config_text,
    find_labelling_file,
    load_config,
)

# Small sets first, VOA last.
SETS = ("teacher", "labels", "common_voice", "fleurs", "opentts", "tg_voices", "cv10_test", "ukr_dialects", "voa")
PUBLIC = {
    "opentts": ("paths.opentts", ["voices/*/train.jsonl", "data/*.zip"]),
    "tg_voices": ("paths.tg_voices", ["data/*.parquet"]),
    "ukr_dialects": ("paths.ukr_dialects", ["data/*.parquet"]),
    "cv10_test": ("paths.cv10_test", ["data/*.parquet"]),
    "voa": ("pseudo_labelling.voa", ["data/train-*.parquet"]),
}
PRIVATE_FOLDERS = {
    "common_voice": "common-voice",
    "fleurs": "fleurs-uk",
    "teacher": "checkpoints",
    "labels": "pseudo-labelling",
}
PLACEHOLDER = "CHANGE_ME"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=SETS, help="Download only these sets.")
    parser.add_argument("--skip", nargs="+", choices=SETS, default=[], help="Do not download these sets.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep Common Voice tar files after extracting.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel file downloads per repo.")
    return parser.parse_args()


def load_token() -> None:
    """Fill HF_TOKEN from .env when the environment does not have it."""
    if os.environ.get("HF_TOKEN"):
        return
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        value = value.strip().strip("\"'")
        # .env.example used to call it HF-TOKEN.
        if separator and key.strip() in {"HF_TOKEN", "HF-TOKEN"} and value:
            os.environ["HF_TOKEN"] = value
            return


def note(name: str, message: str) -> None:
    print(f"[{name}] {message}", flush=True)


def merge_move(source: Path, destination: Path) -> None:
    """Move ``source`` into ``destination``. Files already at the destination stay."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return
    if source.is_dir() and destination.is_dir():
        for child in list(source.iterdir()):
            merge_move(child, destination / child.name)
        if not any(source.iterdir()):
            source.rmdir()
        return
    print(f"  keep existing {destination}", flush=True)


class Downloader:
    def __init__(self, args: argparse.Namespace) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        self.snapshot_download = snapshot_download
        self.access_errors = (GatedRepoError, RepositoryNotFoundError)
        self.args = args
        self.config = load_config()
        self.staging = config_file(self.config, "paths.checkpoints").parent / ".hf-staging"

    def repo(self, key: str) -> str:
        repo_id = config_text(self.config, f"downloads.{key}")
        if PLACEHOLDER in repo_id:
            raise SystemExit(
                f"{CONFIG_PATH.name} downloads.{key} is still {repo_id}. "
                "Set it to your private repo (see HF_UPLOAD.md)."
            )
        return repo_id

    def fetch(self, name: str, repo_id: str, local_dir: Path, patterns: list[str]) -> None:
        note(name, f"{repo_id} -> {local_dir}")
        try:
            self.snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=patterns,
                max_workers=self.args.workers,
            )
        except self.access_errors as error:
            raise SystemExit(
                f"[{name}] no access to {repo_id}: {type(error).__name__}. Check HF_TOKEN; for a gated repo "
                f"request access at https://huggingface.co/datasets/{repo_id} with the same account."
            ) from None

    def place(self, name: str, target: Path) -> None:
        """Move a downloaded private folder to ``target`` and drop what is left in staging."""
        downloaded = self.fetch_private(name)
        merge_move(downloaded, target)
        # Files that were already at the target stay in staging; they are not needed.
        shutil.rmtree(downloaded, ignore_errors=True)

    def fetch_private(self, name: str) -> Path:
        """Download one folder of the private repo into staging. Returns that folder."""
        folder = PRIVATE_FOLDERS[name]
        pattern = f"{folder}/*.tar*" if name == "common_voice" else f"{folder}/**"
        self.fetch(name, self.repo("private_repo"), self.staging, [pattern])
        downloaded = self.staging / folder
        if not downloaded.is_dir():
            raise SystemExit(f"[{name}] {folder}/ is empty in {self.repo('private_repo')}. See HF_UPLOAD.md.")
        return downloaded

    def public(self, name: str) -> None:
        key, patterns = PUBLIC[name]
        target = config_file(self.config, key)
        self.fetch(name, self.repo(name), target, patterns)
        if name == "voa":
            student_voa = config_file(self.config, "student.voa")
            if student_voa.resolve() != target.resolve():
                note(name, f"student.voa is {student_voa}, not {target}; point both at the same directory")

    def teacher(self) -> None:
        teacher = config_file(self.config, "pseudo_labelling.teacher_model")
        if teacher.is_file():
            note("teacher", f"present: {teacher}")
            return
        self.place("teacher", config_file(self.config, "paths.checkpoints"))
        if not teacher.is_file():
            raise SystemExit(f"[teacher] {teacher} is still missing. Check checkpoints/ in the private repo.")
        note("teacher", f"ready: {teacher}")

    def labels(self) -> None:
        output_dir = config_file(self.config, "pseudo_labelling.output_dir")
        teacher = config_file(self.config, "pseudo_labelling.teacher_model")
        found = find_labelling_file(output_dir, teacher, "labels.csv")
        if found is not None:
            note("labels", f"present: {found}")
            return
        self.place("labels", output_dir)
        found = find_labelling_file(output_dir, teacher, "labels.csv")
        if found is None:
            raise SystemExit(
                f"[labels] no labels.csv for {teacher.name} under {output_dir}. The folder in the private repo "
                f"must be pseudo-labelling/{teacher.stem}-NN/."
            )
        note("labels", f"ready: {found}")

    def fleurs(self) -> None:
        target = config_file(self.config, "paths.fleurs")
        splits = ("train", "validation", "test")
        if all((target / split / "state.json").is_file() for split in splits):
            note("fleurs", f"present: {target}")
            return
        self.place("fleurs", target)
        missing = [split for split in splits if not (target / split / "state.json").is_file()]
        if missing:
            raise SystemExit(f"[fleurs] {target} has no state.json for: {', '.join(missing)}")
        note("fleurs", f"ready: {target}")

    def common_voice(self) -> None:
        target = config_file(self.config, "paths.common_voice")
        clips = target / "clips"
        if clips.is_dir() and any(target.glob(".extracted-*")):
            note("common_voice", f"present: {clips}")
            return
        downloaded = self.fetch_private("common_voice")
        archives = sorted(path for path in downloaded.iterdir() if path.is_file())
        target.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            marker = target / f".extracted-{archive.name}"
            if not marker.is_file():
                note("common_voice", f"extract {archive.name} -> {target}")
                with tarfile.open(archive, "r:*") as bundle:
                    bundle.extractall(target, filter="data")
                marker.touch()
            if not self.args.keep_archives:
                archive.unlink()
        if not any(downloaded.iterdir()):
            downloaded.rmdir()
        if not clips.is_dir():
            raise SystemExit(f"[common_voice] {clips} is missing. The archive must contain clips/ at its top level.")
        note("common_voice", f"ready: {clips}")

    def run(self, name: str) -> None:
        if name in PUBLIC:
            self.public(name)
        else:
            getattr(self, name)()


def main() -> None:
    args = parse_args()
    load_token()
    selected = [name for name in SETS if (args.only is None or name in args.only) and name not in args.skip]
    print(f"config {CONFIG_PATH}", flush=True)
    print(f"sets {' '.join(selected) or '(none)'}", flush=True)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set: the private repo and gated VOA will not download", flush=True)
    downloader = Downloader(args)
    for name in selected:
        downloader.run(name)
    if downloader.staging.is_dir() and not any(path.name != ".cache" for path in downloader.staging.iterdir()):
        shutil.rmtree(downloader.staging)
    print("done", flush=True)


if __name__ == "__main__":
    main()
