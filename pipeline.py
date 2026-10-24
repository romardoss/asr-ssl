"""Run the existing Whisper blocks in order, from ``config.yaml``.

Language filtering and 30-second splits are skipped when their files are
already present and the stage is ``skip_if_present``. The same switch covers
pseudo-labelling (``labels.csv`` in the newest run) and the judge (``label_marks.csv``).
Student tuning and evaluation are ``run`` or ``skip``.

    python pipeline.py
    python pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
LANGUAGE_CSV = ROOT / "datasets" / "voa-ukr" / "ukrainian.csv"
TIMESTAMPS_CSV = ROOT / "datasets" / "voa-ukr" / "ukrainian_30s.csv"
WHISPER_META = ROOT / "datasets" / "voa-ukr" / "voa_ukr_meta.json"
MMS_CSV = ROOT / "datasets" / "voa-ukr" / "mms-language-confidences.csv"
FILE_MODES = {"run", "skip", "skip_if_present"}
TAIL_MODES = {"run", "skip"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ft_settings = _load("fine_tune_settings", ROOT / "fine-tune" / "settings.py")
labelling_settings = _load("labelling_settings", ROOT / "pseudo-labelling" / "settings.py")


@dataclass(frozen=True)
class PipelineSettings:
    language: str
    timestamps: str
    label: str
    clean: str
    student: str
    evaluate: str
    chunk_processes: int
    mms_model: str
    mms_batch_size: int
    mms_fp16: bool
    judge_workers: int
    judge_retries: int
    judge_model: str
    filter_marks: bool
    labels_spec: str
    checkpoints: Path
    student_name: str


def _mapping(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"config.yaml: '{name}' must be a mapping")
    return value


def _require(section: dict, key: str, label: str):
    if key not in section:
        raise SystemExit(f"config.yaml missing {label}")
    return section[key]


def _mode(section: dict, key: str, allowed: set[str]) -> str:
    label = f"pipeline.{key}"
    value = _require(section, key, label)
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise SystemExit(f"config.yaml {label} must be one of: {choices}")
    return value


def _text(section: dict, key: str, label: str) -> str:
    value = _require(section, key, label)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"config.yaml {label} must be a non-empty string")
    return value.strip()


def _bool(section: dict, key: str, label: str) -> bool:
    value = _require(section, key, label)
    if not isinstance(value, bool):
        raise SystemExit(f"config.yaml {label} must be true or false")
    return value


def _positive_int(section: dict, key: str, label: str) -> int:
    value = _require(section, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"config.yaml {label} must be a positive integer")
    return value


def load_pipeline(path: Path | None = None) -> PipelineSettings:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{config_path} must be a mapping")
    section = _mapping(loaded, "pipeline")
    student = _mapping(loaded, "student")
    paths = _mapping(loaded, "paths")
    judge_model = _require(section, "judge_model", "pipeline.judge_model")
    if not isinstance(judge_model, str):
        raise SystemExit("config.yaml pipeline.judge_model must be a string")
    return PipelineSettings(
        language=_mode(section, "language", FILE_MODES),
        timestamps=_mode(section, "timestamps", FILE_MODES),
        label=_mode(section, "label", FILE_MODES),
        clean=_mode(section, "clean", FILE_MODES),
        student=_mode(section, "student", TAIL_MODES),
        evaluate=_mode(section, "evaluate", TAIL_MODES),
        chunk_processes=_positive_int(section, "chunk_processes", "pipeline.chunk_processes"),
        mms_model=_text(section, "mms_model", "pipeline.mms_model"),
        mms_batch_size=_positive_int(section, "mms_batch_size", "pipeline.mms_batch_size"),
        mms_fp16=_bool(section, "mms_fp16", "pipeline.mms_fp16"),
        judge_workers=_positive_int(section, "judge_workers", "pipeline.judge_workers"),
        judge_retries=_positive_int(section, "judge_retries", "pipeline.judge_retries"),
        judge_model=judge_model.strip(),
        filter_marks=_bool(student, "filter_marks", "student.filter_marks"),
        labels_spec=_text(student, "labels", "student.labels"),
        checkpoints=Path(_text(paths, "checkpoints", "paths.checkpoints")),
        student_name=_text(student, "name", "student.name"),
    )


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def should_run(mode: str, ready: bool) -> bool:
    if mode == "run":
        return True
    if mode == "skip":
        return False
    return not ready


def require_script_paths(labelling) -> None:
    """The prep scripts write fixed files. Config has to name those same files."""
    expected = {
        "pseudo_labelling.ukrainian_meta": (labelling.ukrainian_meta, LANGUAGE_CSV),
        "pseudo_labelling.timestamps": (labelling.timestamps, TIMESTAMPS_CSV),
    }
    for label, (configured, canonical) in expected.items():
        if configured.resolve() != canonical.resolve():
            raise SystemExit(
                f"config.yaml {label} is {configured}, but the prep scripts write {canonical}."
            )


def execute(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Command exited with status {completed.returncode}: {Path(command[1]).name}")


def note(stage: str, action: str, detail: str) -> None:
    print(f"[{stage}] {action}: {detail}", flush=True)


def label_ready(output_dir: Path, teacher: Path) -> bool:
    """True when the newest teacher run already has a labels file.

    ``label: run`` still calls ``label.py``, which resumes an unfinished file
    or opens the next numbered directory when that file is complete.
    """
    runs = ft_settings.labelling_runs(output_dir, teacher)
    return bool(runs) and present(runs[-1] / "labels.csv")


def resolve_labels(spec: str, labelling) -> Path | None:
    if spec == "latest":
        return ft_settings.find_labelling_file(labelling.output_dir, labelling.teacher_model, "labels.csv")
    path = Path(spec)
    if not path.is_absolute():
        path = ROOT / path
    return path


def run_language(cfg: PipelineSettings, env: dict[str, str], dry_run: bool) -> None:
    py = sys.executable
    execute([py, str(ROOT / "scripts" / "monitor" / "meta_collector.py")], env, dry_run)
    command = [
        py,
        str(ROOT / "scripts" / "monitor" / "predict_languages.py"),
        "--output",
        str(MMS_CSV),
        "--meta",
        str(WHISPER_META),
        "--root",
        env["VOA_UKR_ROOT"],
        "--model-id",
        cfg.mms_model,
        "--batch-size",
        str(cfg.mms_batch_size),
    ]
    if cfg.mms_fp16:
        command.append("--fp16")
    execute(command, env, dry_run)
    execute([py, str(ROOT / "scripts" / "monitor" / "split_language_meta.py")], env, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print each stage without starting it.")
    args = parser.parse_args()

    cfg = load_pipeline()
    labelling = labelling_settings.load_labelling()
    require_script_paths(labelling)
    env = os.environ.copy()
    env["VOA_UKR_ROOT"] = str(labelling.voa)
    py = sys.executable

    if cfg.language == "skip" and not present(labelling.ukrainian_meta):
        raise SystemExit(f"pipeline.language is skip, but {labelling.ukrainian_meta} is missing.")
    if cfg.timestamps == "skip" and not present(labelling.timestamps):
        raise SystemExit(f"pipeline.timestamps is skip, but {labelling.timestamps} is missing.")

    if should_run(cfg.language, present(labelling.ukrainian_meta)):
        note("language", "run", f"write {labelling.ukrainian_meta}")
        run_language(cfg, env, args.dry_run)
        if not args.dry_run and not present(labelling.ukrainian_meta):
            raise SystemExit(f"Language split did not write {labelling.ukrainian_meta}")
    else:
        note("language", "skip", str(labelling.ukrainian_meta))

    if should_run(cfg.timestamps, present(labelling.timestamps)):
        if not present(labelling.ukrainian_meta):
            raise SystemExit(f"Time splits need {labelling.ukrainian_meta}")
        note("timestamps", "run", f"write {labelling.timestamps}")
        execute(
            [
                py,
                str(ROOT / "pseudo-labelling" / "vad-chunking" / "chunk_orchestrator.py"),
                "--max-processes",
                str(cfg.chunk_processes),
            ],
            env,
            args.dry_run,
        )
        if not args.dry_run and not present(labelling.timestamps):
            raise SystemExit(f"Time split did not write {labelling.timestamps}")
    else:
        note("timestamps", "skip", str(labelling.timestamps))

    finished = label_ready(labelling.output_dir, labelling.teacher_model)
    labelling_now = should_run(cfg.label, finished)
    if labelling_now:
        if not present(labelling.ukrainian_meta) or not present(labelling.timestamps):
            raise SystemExit("Pseudo-labelling needs the language list and the time splits.")
        note("label", "run", str(labelling.output_dir))
        execute([py, str(ROOT / "pseudo-labelling" / "label.py")], env, args.dry_run)
    else:
        note("label", "skip", "newest labelling run already has labels.csv")

    labels = resolve_labels(cfg.labels_spec, labelling)
    labels_ready = labels is not None and present(labels)
    if not labels_ready and not (args.dry_run and labelling_now):
        raise SystemExit("No labels.csv for student.labels. Run pipeline.label, or set that path.")
    marks = None if labels is None or not labels_ready else labels.parent / "label_marks.csv"
    marks_ready = marks is not None and present(marks)

    if should_run(cfg.clean, marks_ready):
        if not cfg.judge_model:
            raise SystemExit("pipeline.judge_model is empty. Set it before running the judge.")
        note("clean", "run", str(marks) if marks is not None else "label_marks.csv beside the new labels")
        if labels_ready and marks is not None:
            judge_env = dict(env)
            judge_env["OPENROUTER_MODEL"] = cfg.judge_model
            execute(
                [
                    py,
                    str(ROOT / "pseudo-labelling" / "llm-as-a-judge" / "judge.py"),
                    "--labels",
                    str(labels),
                    "--workers",
                    str(cfg.judge_workers),
                    "--retries",
                    str(cfg.judge_retries),
                ],
                judge_env,
                args.dry_run,
            )
    else:
        note("clean", "skip", str(marks) if marks is not None else "no marks file")

    clean_runs = should_run(cfg.clean, marks_ready)
    if (
        cfg.student == "run"
        and cfg.filter_marks
        and not marks_ready
        and not clean_runs
        and not (args.dry_run and labelling_now)
    ):
        raise SystemExit(
            "student.filter_marks is true, but label_marks.csv is missing. "
            "Run pipeline.clean, or set student.filter_marks to false."
        )

    if cfg.student == "run":
        note("student", "run", cfg.student_name)
        execute([py, str(ROOT / "fine-tune" / "fine-tune-student.py")], env, args.dry_run)
    else:
        note("student", "skip", cfg.student_name)

    checkpoint = cfg.checkpoints / cfg.student_name / "last.ckpt"
    if cfg.evaluate == "run":
        if not args.dry_run and not present(checkpoint):
            raise SystemExit(f"Student checkpoint not found: {checkpoint}")
        note("evaluate", "run", str(checkpoint))
        execute(
            [py, str(ROOT / "evaluation" / "evaluate.py"), "--model", str(checkpoint)],
            env,
            args.dry_run,
        )
    else:
        note("evaluate", "skip", str(checkpoint))


if __name__ == "__main__":
    main()
