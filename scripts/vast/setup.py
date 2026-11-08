"""Налаштування інстансу vast.ai для роботи з цим репозиторієм.

Скрипт готує середовище на свіжому контейнері vast.ai і перевіряє, чи
інстанс придатний для запуску пайплайна. Його можна запускати повторно:
кожен крок пропускається, якщо вже виконаний.

Як запускати (з кореня репозиторію, системним Python образу):

    cd /workspace/asr_ssl
    python3 scripts/vast/setup.py

Той самий рядок можна вставити в On-start Script шаблону vast.ai.

Що робить скрипт, по кроках:

1. Python. Перевіряє, що версія не нижча за 3.11: цього вимагають
   ``pandas==3.0.6`` і ``numpy==2.5.3`` з ``requirements.txt``.
2. Системні пакети. Через ``apt-get`` ставить ``ffmpeg`` (код викликає його
   через ``subprocess`` для декодування аудіо), а також ``tmux``, ``rsync`` і ``git``.
   Пакети, які вже є в ``PATH``, пропускає.
3. Python-пакети. Ставить ``requirements.txt`` через ``pip``. Якщо в образі вже
   є ``torch`` із CUDA, його версія фіксується файлом обмежень (constraints),
   щоб pip не замінив її збіркою під іншу версію CUDA.
4. Файл оточення ``.vast/env.sh`` (у ``.gitignore``). У ньому:

   1. ``ASR_CONFIG``: конфіг, який читають усі скрипти (за замовчуванням
      ``config.vast.yaml``);
   2. ``HF_HOME``: кеш Hugging Face на великому диску ``/workspace``;
   3. ``LD_LIBRARY_PATH``: бібліотеки cuDNN і cuBLAS з pip-пакетів ``nvidia-*``,
      потрібні ``ctranslate2``/``faster-whisper`` на етапі ``language``;
   4. ``PYTHONUNBUFFERED=1``, щоб логи одразу потрапляли у файл.

   До ``~/.bashrc`` додається рядок ``source .vast/env.sh``, тож кожна нова
   SSH-сесія отримує ці змінні. Секрети у файл не пишуться.
5. Теки. Створює ``paths.checkpoints``, ``paths.fine_tune_output``,
   ``paths.evaluation``, ``pseudo_labelling.output_dir`` і ``HF_HOME``.
6. Дані, лише з ``--download``. Запускає ``scripts/vast/download_data.py``:
   публічні датасети з HF і приватний репозиторій ``downloads.private_repo``
   (Common Voice, FLEURS, чекпоінт учителя, ``labels.csv``). Дані
   завантажуються на диск, а не стрімляться. Щоб завантажити лише частину
   наборів, запустіть ``download_data.py`` напряму з ``--only``/``--skip``.
7. Перевірки інстансу. Помилки зупиняють скрипт з ненульовим кодом виходу,
   попередження лише виводяться:

   1. помилка: CUDA недоступна, або в конфігу ввімкнена GPU, якої немає в
      контейнері (на vast.ai нумерація починається з 0);
   2. помилка: конфіг не проходить перевірку ``fine-tune/settings.py`` і
      ``pseudo-labelling/settings.py``;
   3. помилка: немає ``ffmpeg``;
   4. попередження: GPU не підтримує bf16, хоча конфіг його використовує;
   5. попередження: ``/dev/shm`` менше 2 ГБ (воркери ``DataLoader`` можуть
      падати з ``Bus error``) або на диску менше 200 ГБ вільного місця;
   6. попередження: немає датасетів, чекпоінта вчителя або ``labels.csv``.
      Без ``labels.csv`` етап ``label`` заново розмітить приблизно 2000 годин VOA;
   7. попередження: немає ``HF_TOKEN``, або немає ``OPENROUTER_API_KEY``, коли
      ``pipeline.clean`` не ``skip``.

Чого скрипт не робить: не запускає навчання і не вивантажує результати.
Ці кроки описано в ``VAST_AI.md``.

Параметри:

    --config PATH    конфіг для ASR_CONFIG (типово config.vast.yaml)
    --workspace DIR  великий диск інстансу (типово /workspace)
    --skip-apt       не ставити системні пакети
    --skip-pip       не ставити Python-пакети
    --download       завантажити всі датасети й артефакти (крок 6)
    --check-only     лише перевірки, без встановлення й запису файлів
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / ".vast"
ENV_FILE = ENV_DIR / "env.sh"
BASHRC_MARKER = "# asr_ssl vast.ai environment"
APT_PACKAGES = {"ffmpeg": "ffmpeg", "tmux": "tmux", "rsync": "rsync", "git": "git"}
MIN_PYTHON = (3, 11)
MIN_SHM_GB = 2
MIN_FREE_DISK_GB = 200
GB = 1024**3


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"  ok    {message}", flush=True)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  WARN  {message}", flush=True)

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"  ERROR {message}", flush=True)


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def run(command: list[str], **kwargs) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.vast.yaml", help="Config for ASR_CONFIG.")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"), help="Large disk of the instance.")
    parser.add_argument("--skip-apt", action="store_true", help="Do not install system packages.")
    parser.add_argument("--skip-pip", action="store_true", help="Do not install Python packages.")
    parser.add_argument("--download", action="store_true", help="Run scripts/vast/download_data.py.")
    parser.add_argument("--check-only", action="store_true", help="Only run checks.")
    return parser.parse_args()


def check_python(report: Report) -> None:
    step("Python")
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] < MIN_PYTHON:
        report.error(f"Python {version} at {sys.executable}; requirements.txt needs >= 3.11. Use a newer image.")
        raise SystemExit(1)
    report.ok(f"Python {version} at {sys.executable}")


def install_apt() -> None:
    step("System packages")
    missing = [package for binary, package in APT_PACKAGES.items() if shutil.which(binary) is None]
    if not missing:
        print("all present", flush=True)
        return
    prefix: list[str] = []
    if os.geteuid() != 0:
        if shutil.which("sudo") is None:
            raise SystemExit(f"Need root or sudo to install: {' '.join(missing)}")
        prefix = ["sudo"]
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    run(prefix + ["apt-get", "update"], env=env)
    run(prefix + ["apt-get", "install", "-y", "--no-install-recommends", *missing], env=env)


def installed_torch() -> str | None:
    """Version of torch already in the image, when it was built with CUDA."""
    probe = "import torch; print(torch.__version__ if torch.version.cuda else '')"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    version = result.stdout.strip()
    return version if result.returncode == 0 and version else None


def install_pip() -> None:
    step("Python packages")
    command = [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    torch_version = installed_torch()
    if torch_version is not None:
        ENV_DIR.mkdir(exist_ok=True)
        constraints = ENV_DIR / "constraints.txt"
        constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
        print(f"keeping image torch {torch_version}", flush=True)
        command += ["-c", str(constraints)]
    run(command)


def nvidia_library_dirs() -> list[str]:
    """cuDNN and cuBLAS from pip nvidia-* wheels, for ctranslate2."""
    roots = [Path(path) for path in site.getsitepackages()]
    found = []
    for root in roots:
        for name in ("cudnn", "cublas"):
            lib = root / "nvidia" / name / "lib"
            if lib.is_dir() and str(lib) not in found:
                found.append(str(lib))
    return found


def write_env_file(config: str, hf_home: Path) -> None:
    step("Environment file")
    libraries = nvidia_library_dirs()
    lines = [
        "# Written by scripts/vast/setup.py. Re-run the script to refresh.",
        f"export ASR_CONFIG={config}",
        f"export HF_HOME={hf_home}",
        "export PYTHONUNBUFFERED=1",
    ]
    if libraries:
        joined = ":".join(libraries)
        lines.append(f'export LD_LIBRARY_PATH="{joined}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"')
    else:
        print("no pip cuDNN/cuBLAS found; only the language stage (faster-whisper) needs them", flush=True)
    ENV_DIR.mkdir(exist_ok=True)
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ENV_FILE}", flush=True)

    bashrc = Path.home() / ".bashrc"
    current = bashrc.read_text(encoding="utf-8") if bashrc.is_file() else ""
    if BASHRC_MARKER not in current:
        with bashrc.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{BASHRC_MARKER}\n[ -f {ENV_FILE} ] && source {ENV_FILE}\n")
        print(f"added source line to {bashrc}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_config(report: Report):
    """Validate the config with the project's own loaders. Returns (raw config, settings, labelling)."""
    step(f"Config {os.environ['ASR_CONFIG']}")
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    import project_config

    loaded = project_config.load_config()
    try:
        settings = load_module("vast_ft_settings", ROOT / "fine-tune" / "settings.py").load_settings()
        labelling = load_module("vast_label_settings", ROOT / "pseudo-labelling" / "settings.py").load_labelling()
    except SystemExit as error:
        report.error(f"config does not validate: {error}")
        raise SystemExit(1)
    report.ok(f"{project_config.CONFIG_PATH} validates")
    return project_config, loaded, settings, labelling


def make_directories(settings, labelling, hf_home: Path) -> None:
    step("Directories")
    for path in (settings.checkpoints, settings.output_dir, settings.evaluation_dir, labelling.output_dir, hf_home):
        path.mkdir(parents=True, exist_ok=True)
        print(f"ready {path}", flush=True)


def configured_gpus(loaded: dict) -> set[int]:
    indexes = set()
    for key, enabled in loaded.get("gpus", {}).items():
        if enabled is True and key.startswith("nvidia") and key[6:].isdigit():
            indexes.add(int(key[6:]))
    return indexes


def check_gpu(report: Report, loaded: dict) -> None:
    step("GPU")
    probe = (
        "import json, torch\n"
        "count = torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "devices = [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory) for i in range(count)]\n"
        "bf16 = bool(count) and torch.cuda.is_bf16_supported()\n"
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'devices': devices, 'bf16': bf16}))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        report.error(f"torch probe failed: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'no output'}")
        return
    import json

    info = json.loads(result.stdout.strip().splitlines()[-1])
    devices = info["devices"]
    if not devices:
        report.error(f"torch {info['torch']} sees no CUDA GPU")
        return
    report.ok(f"torch {info['torch']}, CUDA {info['cuda']}")
    for index, (name, memory) in enumerate(devices):
        report.ok(f"cuda:{index} {name}, {memory / GB:.0f} GB")
    for index in sorted(configured_gpus(loaded)):
        if index >= len(devices):
            report.error(f"config enables nvidia{index}, but the container has {len(devices)} GPU(s); use nvidia0")
    if not info["bf16"]:
        report.warn("GPU has no bf16 support; config precision bf16 needs Ampere or newer")
    print("  note  batch sizes were tuned on the original server; check them with limit_train_batches", flush=True)


def check_resources(report: Report, workspace: Path) -> None:
    step("Resources")
    if shutil.which("ffmpeg") is None:
        report.error("ffmpeg is not in PATH")
    else:
        report.ok("ffmpeg present")
    shm = Path("/dev/shm")
    if shm.is_dir():
        size = shutil.disk_usage(shm).total / GB
        message = f"/dev/shm {size:.1f} GB"
        if size < MIN_SHM_GB:
            report.warn(f"{message}; lower num_workers or start the container with a larger --shm-size")
        else:
            report.ok(message)
    target = workspace if workspace.is_dir() else ROOT
    free = shutil.disk_usage(target).free / GB
    message = f"{free:.0f} GB free on {target}"
    if free < MIN_FREE_DISK_GB:
        report.warn(f"{message}; all datasets need about {MIN_FREE_DISK_GB} GB or more")
    else:
        report.ok(message)


def check_data(report: Report, project_config, loaded: dict, settings, labelling) -> None:
    step("Data and checkpoints")
    dataset_paths = {
        "common_voice": settings.common_voice / "clips",
        "fleurs": settings.fleurs,
        "opentts": settings.opentts,
        "tg_voices": settings.tg_voices,
        "ukr_dialects": settings.ukr_dialects,
        "cv10_test": settings.cv10_test,
    }
    used = set(settings.train_data) | set(settings.dev_data) | set(settings.test_data) | {"common_voice"}
    for name in sorted(used):
        path = dataset_paths[name]
        if path.is_dir():
            report.ok(f"{name}: {path}")
        else:
            report.warn(f"{name} missing: {path}")
    voa = labelling.voa / "data"
    if any(voa.glob("train-*.parquet")):
        report.ok(f"VOA shards: {voa}")
    else:
        report.warn(f"VOA shards missing: {voa}")
    if labelling.teacher_model.is_file():
        report.ok(f"teacher checkpoint: {labelling.teacher_model}")
    else:
        report.warn(f"teacher checkpoint missing: {labelling.teacher_model}")
    labels = project_config.find_labelling_file(labelling.output_dir, labelling.teacher_model, "labels.csv")
    if labels is not None:
        report.ok(f"labels.csv: {labels}")
    else:
        report.warn(
            f"no labels.csv for {labelling.teacher_model.name} under {labelling.output_dir}; "
            "pipeline.label would relabel all of VOA"
        )


def dotenv_keys() -> set[str]:
    path = ROOT / ".env"
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                keys.add(key.strip())
    return keys


def dotenv_value(name: str) -> str:
    path = ROOT / ".env"
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("\"'")
    return ""


def check_secrets(report: Report, loaded: dict) -> None:
    step("Secrets")
    available = {key for key, value in os.environ.items() if value} | dotenv_keys()
    if "HF_TOKEN" in available:
        report.ok("HF_TOKEN set")
    else:
        report.warn("HF_TOKEN is not set; private or gated Hugging Face repos will not download")
    clean = loaded.get("pipeline", {}).get("clean")
    if clean != "skip":
        if "OPENROUTER_API_KEY" in available:
            report.ok("OPENROUTER_API_KEY set")
        else:
            report.warn(f"pipeline.clean is {clean}, but OPENROUTER_API_KEY is not set")
        switch = os.environ.get("USE_DECISION_INSTEAD_OF_LLM") or dotenv_value("USE_DECISION_INSTEAD_OF_LLM")
        if (switch or "").strip().lower() in {"1", "true", "yes", "on"}:
            if "OPENROUTER_DECISION_MODEL" in available:
                report.ok("USE_DECISION_INSTEAD_OF_LLM: judge uses OPENROUTER_DECISION_MODEL")
            else:
                report.warn("USE_DECISION_INSTEAD_OF_LLM is true, but OPENROUTER_DECISION_MODEL is not set")


def main() -> None:
    args = parse_args()
    report = Report()
    config = args.config
    config_file = Path(config) if Path(config).is_absolute() else ROOT / config
    if not config_file.is_file():
        raise SystemExit(f"Config not found: {config_file}")
    os.environ["ASR_CONFIG"] = config
    workspace = args.workspace
    default_cache = workspace / "hf-cache" if workspace.is_dir() else ENV_DIR / "hf-cache"
    hf_home = Path(os.environ.get("HF_HOME") or default_cache)
    os.environ["HF_HOME"] = str(hf_home)

    check_python(report)
    if not args.check_only:
        if not args.skip_apt:
            install_apt()
        if not args.skip_pip:
            install_pip()
        write_env_file(config, hf_home)

    project_config, loaded, settings, labelling = check_config(report)
    if not args.check_only:
        make_directories(settings, labelling, hf_home)
        if args.download:
            step("Data")
            run([sys.executable, str(ROOT / "scripts" / "vast" / "download_data.py")])
    check_gpu(report, loaded)
    check_resources(report, workspace)
    check_data(report, project_config, loaded, settings, labelling)
    check_secrets(report, loaded)

    step("Summary")
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)", flush=True)
    if report.errors:
        raise SystemExit(1)
    print(
        f"\nNext:\n  source {ENV_FILE}\n  python main.py pipeline --dry-run\n"
        "  tmux new -s train 'python main.py pipeline 2>&1 | tee output/pipeline.log'",
        flush=True,
    )


if __name__ == "__main__":
    main()
