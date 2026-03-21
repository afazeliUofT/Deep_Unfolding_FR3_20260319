from __future__ import annotations

import _repo_bootstrap as _rb

_rb.bootstrap()

import importlib
import json

import tensorflow as tf

from fr3_twc.fer import simulate_5g_nr_fer_curve


def _version(name: str) -> str:
    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", "unknown"))


def main() -> None:
    info = {
        "tensorflow": _version("tensorflow"),
        "sionna": _version("sionna"),
        "num_gpus": len(tf.config.list_physical_devices("GPU")),
    }
    print("SIONNA_PROBE_ENV", json.dumps(info, sort_keys=True))

    curve = simulate_5g_nr_fer_curve(
        sinr_db_points=[0.0],
        modulation_order=2,
        code_rate=0.5,
        k_bits=256,
        num_frames_per_point=16,
        max_frame_errors=8,
        decoder_iterations=10,
        require_sionna=True,
        allow_fallback=False,
    )
    row0 = curve.rows[0]
    print(
        "SIONNA_FER_OK",
        json.dumps(
            {
                "used_sionna": bool(curve.used_sionna),
                "backend": str(curve.backend),
                "sionna_version": str(curve.sionna_version),
                "fer_at_0db": float(row0["fer"]),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
