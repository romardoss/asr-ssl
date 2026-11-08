"""OpenRouter Decisions API as an alternative to the text LLM judge.

A decision model (for example ``typesafe/jev-1.14``) does not write text. It
answers typed questions about a ``state`` with calibrated probabilities. Here
each transcript is one state and one ``score`` question whose ordered levels
are the 0-5 marks of ``pseudo-labelling/llm-as-a-judge/prompt.txt``.

The backend is chosen by the environment (``.env`` fills names that are not set):

    USE_DECISION_INSTEAD_OF_LLM=true      use this module instead of the chat judge
    OPENROUTER_DECISION_MODEL=typesafe/jev-1.14
    OPENROUTER_API_KEY=...                 the same key as the chat judge

The Decisions API is in alpha (``/api/alpha/decisions``); its shape may change.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
SWITCH_ENV = "USE_DECISION_INSTEAD_OF_LLM"
MODEL_ENV = "OPENROUTER_DECISION_MODEL"
LLM_MODEL_ENV = "OPENROUTER_MODEL"
KEY_ENV = "OPENROUTER_API_KEY"
BACKENDS = ("llm", "decisions")
QUESTION_ID = "label_quality"
_LEVEL = re.compile(r"^\s*(\d+)\s*[—–-]\s*(.+?)\s*$")


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


def use_decisions() -> bool:
    """True when ``USE_DECISION_INSTEAD_OF_LLM`` is set to a true value."""
    load_dotenv()
    raw = os.environ.get(SWITCH_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise SystemExit(f"{SWITCH_ENV} must be true or false, got {raw!r}")


def configured_backend() -> str:
    return "decisions" if use_decisions() else "llm"


def model_for(backend: str) -> str:
    """The OpenRouter model id the environment sets for ``backend`` (may be empty)."""
    load_dotenv()
    name = MODEL_ENV if backend == "decisions" else LLM_MODEL_ENV
    return os.environ.get(name, "").strip()


def rubric(prompt: str) -> tuple[str, list[str]]:
    """Split the judge prompt into instructions and the ordered level texts (index = mark)."""
    levels: dict[int, str] = {}
    instructions = []
    for line in prompt.splitlines():
        match = _LEVEL.match(line)
        if match:
            levels[int(match.group(1))] = match.group(2)
        elif line.strip():
            instructions.append(line.strip())
    if not levels or sorted(levels) != list(range(len(levels))):
        raise SystemExit("The judge prompt needs consecutive `<mark> — <description>` lines starting at 0.")
    return "\n".join(instructions), [levels[mark] for mark in sorted(levels)]


@dataclass(frozen=True)
class Decision:
    mark: int  # most probable level
    score: float  # probability-weighted level
    confidence: float
    probabilities: dict[int, float]


def request_body(model: str, instructions: str, levels: list[str], text: str) -> dict:
    return {
        "model": model,
        "state": text,
        "questions": {
            QUESTION_ID: {
                "type": "score",
                "instructions": instructions,
                "criteria": levels,
            }
        },
    }


def parse_answer(payload: dict, level_count: int) -> Decision | None:
    """The score answer, or None for a refusal or a malformed reply."""
    answer = (payload.get("answers") or {}).get(QUESTION_ID) or {}
    if answer.get("type") != "score":
        return None
    raw = answer.get("probabilities") or {}
    try:
        probabilities = {int(level): float(value) for level, value in raw.items()}
        score = float(answer["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not probabilities or any(level < 0 or level >= level_count for level in probabilities):
        return None
    mark = max(probabilities, key=lambda level: (probabilities[level], -level))
    return Decision(mark, score, float(answer.get("confidence") or 0.0), probabilities)


def decide(api_key: str, model: str, instructions: str, levels: list[str], text: str, timeout: int = 120) -> Decision | None:
    """Score one transcript. Network and HTTP errors are raised to the caller."""
    body = json.dumps(request_body(model, instructions, levels, text)).encode()
    request = urllib.request.Request(
        DECISIONS_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return parse_answer(payload, len(levels))
