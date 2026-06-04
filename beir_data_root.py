"""Resolve BEIR dataset root (shared with trng worktree layout).

Priority
--------
1. ``BEIR_DATA_DIR`` environment variable
2. ``<repo>/beir_datasets``
3. Legacy sibling checkouts under ``prime_hotel/benchmark_data/...``
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
_TRNG_DIR = Path(r"c:\Users\wynos\trng")
_LEGACY = r"C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets"


def resolve_beir_root() -> str:
    env = os.environ.get("BEIR_DATA_DIR", "").strip()
    if env:
        return os.path.normpath(env.rstrip("/\\"))
    candidates = [
        _REPO_DIR / "beir_datasets",
        _TRNG_DIR / "beir_datasets",
        _TRNG_DIR / "benchmark_data" / "synthetic" / "prime_hotel" / "beir_datasets",
        _TRNG_DIR.parent / "prime_hotel" / "benchmark_data" / "synthetic" / "prime_hotel" / "beir_datasets",
        Path(_LEGACY),
    ]
    for c in candidates:
        if c.is_dir():
            return str(c.resolve())
    return str((_REPO_DIR / "beir_datasets").resolve())
