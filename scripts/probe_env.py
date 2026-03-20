from __future__ import annotations

import importlib
import importlib.util
import json
import os
import site
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    s = str(path)
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)

try:
    import _repo_bootstrap as _rb  # type: ignore

    ROOT = _rb.bootstrap()
except Exception:
    os.environ.setdefault("FR3_REPO_ROOT", str(ROOT))


def _version(name: str) -> str:
    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", "n/a"))


def _origin(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "NOT_FOUND"
    origin = getattr(spec, "origin", None)
    if origin is None:
        return "namespace"
    return str(origin)


def main() -> None:
    print(f"CWD {Path.cwd()}")
    print(f"PYTHON_EXECUTABLE {sys.executable}")
    print(f"PYTHON_PREFIX {sys.prefix}")
    print(f"REPO_ROOT {ROOT}")
    print(f"PYTHONPATH_ENV {os.environ.get('PYTHONPATH', '')}")
    print(f"FR3_REPO_ROOT_ENV {os.environ.get('FR3_REPO_ROOT', '')}")
    print(
        "REPO_PATHS "
        + json.dumps(
            {
                "root_exists": ROOT.exists(),
                "src_exists": (ROOT / "src").exists(),
                "scripts_exists": (ROOT / "scripts").exists(),
                "fr3_sim_exists": (ROOT / "src" / "fr3_sim").exists(),
                "fr3_twc_exists": (ROOT / "src" / "fr3_twc").exists(),
            },
            sort_keys=True,
        )
    )

    pth_files = []
    for p in site.getsitepackages():
        pth = Path(p) / "fr3_repo_paths.pth"
        if pth.exists():
            pth_files.append(str(pth))
    print("PTH_FILES " + json.dumps(pth_files))
    print("SYS_PATH_HEAD " + json.dumps(sys.path[:12]))

    failed = []
    for name in [
        "fr3_sim",
        "fr3_sim.channel",
        "fr3_sim.topology",
        "fr3_twc",
        "fr3_twc.unfolding",
        "tensorflow",
        "sionna",
    ]:
        try:
            origin = _origin(name)
            version = _version(name.split(".")[0]) if "." not in name else "submodule"
            print(f"IMPORT_OK {name} {version} {origin}")
        except Exception as e:
            print(f"IMPORT_FAIL {name} {type(e).__name__}: {e}")
            failed.append(name)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
