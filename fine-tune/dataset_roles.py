"""Which labelled datasets can fill which role, and their folder names.

Kept apart from ``speech_data.py`` so the config can be checked without
importing torch or the audio decoders. FLEURS and Ukrainian Dialects call the
dev split "validation". The Common Voice 10 test set is stored under the
parquet name "train".
"""

ROLES: dict[str, dict[str, str]] = {
    "common_voice": {"train": "train", "dev": "dev", "test": "test"},
    "fleurs": {"train": "train", "dev": "validation", "test": "test"},
    "opentts": {"train": "train"},
    "tg_voices": {"train": "train"},
    "ukr_dialects": {"train": "train", "dev": "validation", "test": "test"},
    "cv10_test": {"test": "train"},
}


def check_names(names, role: str, label: str) -> None:
    """Stop with a clear message when a name is unknown or has no ``role`` split."""
    for name in names:
        if name not in ROLES:
            known = ", ".join(sorted(ROLES))
            raise SystemExit(f"config.yaml {label}: unknown dataset '{name}'. Known names: {known}")
        if role not in ROLES[name]:
            available = ", ".join(sorted(ROLES[name]))
            raise SystemExit(f"config.yaml {label}: '{name}' has no {role} split (available: {available})")
