#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import selectors
import subprocess
import sys
from pathlib import Path

# Filter only a narrow set of known harmless TensorFlow/Sionna startup lines.
# Everything else passes through unchanged so real failures remain visible.
_HARMLESS_PATTERNS = [
    re.compile(r"Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered"),
    re.compile(r"Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered"),
    re.compile(r"Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered"),
    re.compile(r"TF-TRT Warning: Could not find TensorRT"),
    re.compile(r"This TensorFlow binary is optimized to use available CPU instructions"),
    re.compile(r"To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags"),
    re.compile(r"Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set"),
    re.compile(r"Created device /job:localhost/replica:0/task:0/device:GPU:0"),
    re.compile(r"Creating GpuSolver handles for stream"),
]


def _is_harmless(line: str) -> bool:
    return any(p.search(line) for p in _HARMLESS_PATTERNS)


def _usage() -> int:
    sys.stderr.write(
        "usage: run_quiet_python.py <script.py> [args ...]\n"
    )
    return 2


def main() -> int:
    if len(sys.argv) < 2:
        return _usage()

    target = sys.argv[1]
    target_args = sys.argv[2:]
    cmd = [sys.executable, "-u", target, *target_args]

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    while sel.get_map():
        for key, _ in sel.select():
            stream = key.fileobj
            label = key.data
            line = stream.readline()
            if line == "":
                try:
                    sel.unregister(stream)
                except Exception:
                    pass
                continue
            if label == "stdout":
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                if not _is_harmless(line):
                    sys.stderr.write(line)
                    sys.stderr.flush()

    return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
