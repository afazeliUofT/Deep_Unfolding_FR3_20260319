from __future__ import annotations

try:
    import _repo_bootstrap as _rb  # type: ignore

    _rb.bootstrap()
except Exception:
    pass
