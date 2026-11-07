"""Fine-tune in several steps, each starting from the previous step's weights.

The chain is the ``multi_step`` section of the config. Each step is either

- ``data: pseudo``: VOA slices with their pseudo-labels, trained by
  ``fine-tune-student.py`` with the ``student`` settings. ``accepted_mark``
  keeps only labels with those judge marks; without it every label is used.
- ``data: [common_voice, ...]``: labelled datasets, trained by
  ``fine-tune.py`` with the ``training`` settings and ``data.dev``.
- ``data: [pseudo, common_voice, ...]``: both in one run of
  ``fine-tune-student.py`` (``student.extra_train``): one model, one optimizer.
  Each epoch first goes through the pseudo-labels, then through the labelled
  sets. ``accepted_mark`` filters only the pseudo-labels.

Step 1 starts from ``multi_step.init``, every later step from the final
checkpoint of the step before it. A step may override ``max_epochs``,
``learning_rate``, and ``warmup_steps``; the optimizer and LR schedule start
fresh in each step.

For each step a full config is written to
``output/multi-step/<name>/step<N>.yaml`` and the training script runs with
it, so the checkpoint folder keeps the exact config of its step. Checkpoints
go to ``<paths.checkpoints>/<name>-s<N>-<data>_01``. A step whose final
epoch is already saved is skipped, so running the chain again after a crash
continues where it stopped. To start the same chain over, change
``multi_step.name``.

``multi_step.evaluate`` decides what ``evaluate.py`` scores and adds to
``results.csv`` (with the ``chain`` and ``step`` columns):
``steps`` the final checkpoint of each step, ``all`` also the first epoch of
each step, ``last`` only the end of the chain, ``none`` nothing.

    python fine-tune/multi_step.py --dry-run
    python fine-tune/multi_step.py
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FINE_TUNE = Path(__file__).resolve().parent
ROOT = FINE_TUNE.parent
if str(FINE_TUNE) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE))

from checkpoints import is_final_epoch_checkpoint, latest_checkpoint, student_checkpoint_dirs  # noqa: E402
from run_meta import CHAIN_ENV, NOTE_ENV, STEP_ENV  # noqa: E402
from dataset_roles import check_names  # noqa: E402
from settings import load_settings, load_student_settings  # noqa: E402

from project_config import CONFIG_PATH, load_config  # noqa: E402

PSEUDO = "pseudo"
EVALUATE_MODES = ("steps", "all", "last", "none")
OVERRIDES = {"max_epochs": int, "learning_rate": float, "warmup_steps": int}
OUTPUT_DIR = ROOT / "output" / "multi-step"
STUDENT_SCRIPT = FINE_TUNE / "fine-tune-student.py"
FINE_TUNE_SCRIPT = FINE_TUNE / "fine-tune.py"
EVALUATE_SCRIPT = ROOT / "evaluation" / "evaluate_training.py"


@dataclass(frozen=True)
class Step:
    number: int
    pseudo: bool
    datasets: tuple[str, ...] = ()
    marks: tuple[int, ...] | None = None
    overrides: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.pseudo:
            return "-".join(self.datasets)
        pseudo = "pseudo-m" + "".join(str(mark) for mark in self.marks) if self.marks else "pseudo-all"
        return "-".join([pseudo, *self.datasets])

    @property
    def description(self) -> str:
        if not self.pseudo:
            return ", ".join(self.datasets)
        marks = ", ".join(str(mark) for mark in self.marks) if self.marks else "all"
        return " + ".join([f"pseudo-labels (marks {marks})", *self.datasets])


@dataclass(frozen=True)
class Chain:
    name: str
    init: str
    evaluate: str
    steps: tuple[Step, ...]

    def step_name(self, step: Step) -> str:
        return f"{self.name}-s{step.number}-{step.label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the steps and commands without running them.")
    return parser.parse_args()


def _fail(message: str) -> None:
    raise SystemExit(f"{CONFIG_PATH.name} multi_step: {message}")


def _marks(raw, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        _fail(f"{label} must be a non-empty list of marks from 0 to 5")
    marks: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 5:
            _fail(f"{label} must be a non-empty list of marks from 0 to 5")
        if item not in marks:
            marks.append(item)
    return tuple(sorted(marks))


def _step(number: int, raw) -> Step:
    label = f"steps[{number}]"
    if not isinstance(raw, dict) or "data" not in raw:
        _fail(f"{label} must be a mapping with 'data'")
    unknown = set(raw) - {"data", "accepted_mark", *OVERRIDES}
    if unknown:
        _fail(f"{label} has unknown keys: {', '.join(sorted(unknown))}")
    overrides = {}
    for key, kind in OVERRIDES.items():
        if key in raw:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (kind is int and value != int(value)):
                _fail(f"{label}.{key} must be a non-negative {kind.__name__}")
            if key != "warmup_steps" and value <= 0:
                _fail(f"{label}.{key} must be positive")
            overrides[key] = kind(value)
    data = [raw["data"]] if isinstance(raw["data"], str) else raw["data"]
    if not isinstance(data, list) or not data or not all(isinstance(item, str) and item for item in data):
        _fail(f"{label}.data must be '{PSEUDO}' or a non-empty list of '{PSEUDO}' and dataset names")
    if len(set(data)) != len(data):
        _fail(f"{label}.data lists something twice")
    datasets = tuple(item for item in data if item != PSEUDO)
    check_names(datasets, "train", f"multi_step {label}.data")
    if PSEUDO in data:
        marks = _marks(raw["accepted_mark"], f"{label}.accepted_mark") if "accepted_mark" in raw else None
        return Step(number, True, datasets=datasets, marks=marks, overrides=overrides)
    if "accepted_mark" in raw:
        _fail(f"{label}.accepted_mark applies only when data includes {PSEUDO}")
    return Step(number, False, datasets=datasets, overrides=overrides)


def load_chain(config: dict) -> Chain:
    section = config.get("multi_step")
    if not isinstance(section, dict):
        _fail("section is missing")
    name = section.get("name")
    if not isinstance(name, str) or not name.startswith("student_"):
        _fail("name must be a string that starts with student_")
    init = section.get("init")
    if not isinstance(init, str) or not init.strip():
        _fail("init must be a checkpoint path or a model id")
    evaluate = section.get("evaluate", "steps")
    if evaluate not in EVALUATE_MODES:
        _fail(f"evaluate must be one of: {', '.join(EVALUATE_MODES)}")
    raw_steps = section.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        _fail("steps must be a non-empty list")
    steps = tuple(_step(number, raw) for number, raw in enumerate(raw_steps, start=1))
    return Chain(name.strip(), init.strip(), evaluate, steps)


def step_config(config: dict, chain: Chain, step: Step, init: str) -> dict:
    """The full config one step trains with."""
    derived = copy.deepcopy(config)
    name = chain.step_name(step)
    if step.pseudo:
        student = derived["student"]
        student.update(
            {"name": name, "init": init, "filter_marks": step.marks is not None, "extra_train": list(step.datasets)}
        )
        if step.marks is not None:
            student["accepted_mark"] = list(step.marks)
        student.update(step.overrides)
    else:
        derived["model"]["name"] = name
        derived["data"]["train"] = list(step.datasets)
        training = derived["training"]
        training["init"] = init
        training.update(step.overrides)
    derived.pop("multi_step", None)
    return derived


def max_epochs(derived: dict, step: Step) -> int:
    return int(derived["student" if step.pseudo else "training"]["max_epochs"])


def step_state(checkpoints: Path, name: str, epochs: int) -> tuple[Path | None, Path | None, bool]:
    """Newest checkpoint folder of a step, its newest checkpoint, and whether the step finished."""
    for directory in reversed(student_checkpoint_dirs(checkpoints, name)):
        checkpoint = latest_checkpoint(directory)
        if checkpoint is not None:
            return directory, checkpoint.resolve(), is_final_epoch_checkpoint(checkpoint, epochs)
    return None, None, False


def write_config(chain: Chain, step: Step, derived: dict) -> Path:
    path = OUTPUT_DIR / chain.name / f"step{step.number}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Written by fine-tune/multi_step.py from {CONFIG_PATH.name}: "
        f"chain {chain.name}, step {step.number}/{len(chain.steps)} ({step.description}).\n"
    )
    path.write_text(header + yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def step_note(chain: Chain, step: Step) -> str:
    note = os.environ.get(NOTE_ENV, "").strip()
    where = f"{chain.name} step {step.number}/{len(chain.steps)}: {step.description}"
    return f"{note} | {where}" if note else where


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{Path(command[1]).name} exited with status {completed.returncode}")


def main() -> None:
    args = parse_args()
    config = load_config()
    chain = load_chain(config)
    checkpoints = load_settings().checkpoints
    print(f"chain {chain.name}: {len(chain.steps)} steps from {chain.init}", flush=True)

    # Check every step's config before the first one trains.
    for step in chain.steps:
        derived = step_config(config, chain, step, chain.init)
        path = write_config(chain, step, derived)
        (load_student_settings if step.pseudo else load_settings)(path)

    previous = chain.init
    for step in chain.steps:
        name = chain.step_name(step)
        derived = step_config(config, chain, step, previous)
        path = write_config(chain, step, derived)
        epochs = max_epochs(derived, step)
        env = dict(os.environ)
        env.update({"ASR_CONFIG": str(path), CHAIN_ENV: chain.name, STEP_ENV: str(step.number), NOTE_ENV: step_note(chain, step)})
        print(f"\n== step {step.number}/{len(chain.steps)}: {step.description} -> {name}", flush=True)

        directory, checkpoint, finished = step_state(checkpoints, name, epochs)
        if finished:
            print(f"done already: {checkpoint}", flush=True)
        else:
            if checkpoint is not None:
                print(f"resuming from {checkpoint}", flush=True)
            script = STUDENT_SCRIPT if step.pseudo else FINE_TUNE_SCRIPT
            run([sys.executable, str(script)], env, args.dry_run)
            if args.dry_run:
                previous = f"<final checkpoint of step {step.number}>"
                continue
            directory, checkpoint, finished = step_state(checkpoints, name, epochs)
            if not finished:
                raise SystemExit(f"step {step.number} did not save its final epoch ({epochs}) under {checkpoints}/{name}_NN")

        last = step.number == len(chain.steps)
        if chain.evaluate in {"steps", "all"} or (chain.evaluate == "last" and last):
            command = [sys.executable, str(EVALUATE_SCRIPT), "--checkpoint-dir", str(directory)]
            if chain.evaluate != "all":
                command.append("--final-only")
            # Score with the main config: data.test and model.id stay those of the project.
            run(command, {**env, "ASR_CONFIG": str(CONFIG_PATH)}, args.dry_run)
        previous = str(checkpoint)

    print(f"\nchain {chain.name} finished: {previous}", flush=True)


if __name__ == "__main__":
    main()
