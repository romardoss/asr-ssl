"""Read Common Voice splits and write manifests under fine-tune/manifests.

Uses ``PATH_TO_COMMON_VOICE`` from the environment or the repo ``.env``, and
the train/dev/test TSVs in ``meta/common-voice``.

    python fine-tune/read_common_voice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from common_voice import common_voice_root, load_splits

MANIFEST_DIR = FINE_TUNE_DIR / "manifests"


def main() -> None:
    root = common_voice_root()
    splits = load_splits()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"clips: {root / 'clips'}")
    for name, frame in splits.items():
        destination = MANIFEST_DIR / f"{name}.csv"
        frame.to_csv(destination, index=False)
        print(f"{name}: {len(frame)} clips -> {destination}")
        print(frame.head(2).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
