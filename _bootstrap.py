"""Put ``src/`` on ``sys.path`` so root scripts work before ``pip install -e .``."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src() -> None:
    src = Path(__file__).resolve().parent / "src"
    entry = str(src)
    if src.is_dir() and entry not in sys.path:
        sys.path.insert(0, entry)
