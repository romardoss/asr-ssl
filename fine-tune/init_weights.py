"""Load the starting weights of a training run."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration


def load_init_model(init: str, base: str) -> WhisperForConditionalGeneration:
    """Whisper from ``init``.

    A ``.ckpt`` file is a Lightning checkpoint of this project: its ``model.*``
    weights are loaded on top of ``base``. Anything else is a Hugging Face id
    or a model directory.
    """
    path = Path(init)
    if path.is_file() and path.suffix == ".ckpt":
        model = WhisperForConditionalGeneration.from_pretrained(base)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        weights = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        if not weights:
            raise SystemExit(f"No model weights found in {path}")
        model.load_state_dict(weights)
        return model
    if init.endswith(".ckpt"):
        raise SystemExit(f"Init checkpoint not found: {path}")
    return WhisperForConditionalGeneration.from_pretrained(init)
