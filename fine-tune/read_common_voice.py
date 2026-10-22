"""Read Common Voice splits and write manifests under datasets/common-voice.

Uses ``paths.common_voice`` from ``config.yaml`` and the train/dev/test/unlabelled
TSVs in ``datasets/common-voice/raw``. Each TSV line stays one clip.
Sentences that contain a comma or a quote are written with normal CSV quoting.

    python fine-tune/read_common_voice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from common_voice import common_voice_root, load_splits

MANIFEST_DIR = FINE_TUNE_DIR.parent / "datasets" / "common-voice"


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
