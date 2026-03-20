from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _prepend(path: Path) -> None:
    s = str(path)
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)


def _origin_matches(module: ModuleType, expected_init: Path) -> bool:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin is None:
        return False
    try:
        return Path(origin).resolve() == expected_init.resolve()
    except Exception:
        return str(origin) == str(expected_init)


def _load_package_from_dir(name: str, package_dir: Path) -> ModuleType:
    init_py = package_dir / "__init__.py"
    if not init_py.exists():
        raise ModuleNotFoundError(
            f"Local package '{name}' is missing expected file: {init_py}"
        )

    existing = sys.modules.get(name)
    if existing is not None and _origin_matches(existing, init_py):
        return existing

    spec = importlib.util.spec_from_file_location(
        name,
        str(init_py),
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {name} from {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_local_package(name: str, root: Path) -> ModuleType:
    package_dir = root / "src" / name
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    return _load_package_from_dir(name, package_dir)


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / "src", root / "scripts"):
        _prepend(path)

    os.environ.setdefault("FR3_REPO_ROOT", str(root))

    # Force local packages into sys.modules if normal import resolution is still flaky
    # on the cluster. This directly addresses the observed Narval failure:
    # ModuleNotFoundError: No module named 'fr3_sim'.
    _ensure_local_package("fr3_sim", root)
    _ensure_local_package("fr3_twc", root)
    return root


ROOT = bootstrap()
