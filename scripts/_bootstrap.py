from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
