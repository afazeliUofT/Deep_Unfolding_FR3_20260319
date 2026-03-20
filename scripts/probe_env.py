from __future__ import annotations

import importlib

import _bootstrap  # noqa: F401

def _version(name: str) -> str:
    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", "n/a"))

def main() -> None:
    for name in ["fr3_sim", "fr3_twc", "tensorflow", "sionna"]:
        print(f"IMPORT_OK {name} {_version(name)}")

if __name__ == "__main__":
    main()
