from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> Path:
    """Add repo-local source paths to sys.path and return repo root."""
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / "src", root / "scripts"):
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("FR3_REPO_ROOT", str(root))
    return root


ROOT = bootstrap()
