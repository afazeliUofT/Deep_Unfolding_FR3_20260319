from __future__ import annotations

import os

try:
    import _repo_bootstrap as _rb

    _rb.bootstrap()
    if os.environ.get("FR3_DEBUG_BOOTSTRAP", "0") == "1":
        print(f"SITECUSTOMIZE_BOOTSTRAP_OK {_rb.ROOT}")
except Exception as exc:  # pragma: no cover
    if os.environ.get("FR3_DEBUG_BOOTSTRAP", "0") == "1":
        print(f"SITECUSTOMIZE_BOOTSTRAP_FAIL {type(exc).__name__}: {exc}")
