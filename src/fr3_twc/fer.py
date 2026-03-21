from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FerCurveResult:
    modulation_order: int
    code_rate: float
    k_bits: int
    n_bits: int
    used_sionna: bool
    backend: str
    sionna_version: str
    error_message: str
    rows: List[Dict[str, Any]]


def _import_sionna_blocks() -> Dict[str, Any]:
    errs: list[str] = []

    try:
        import sionna  # type: ignore
        from sionna.phy.utils import BinarySource  # type: ignore
        from sionna.phy.channel.awgn import AWGN  # type: ignore
        from sionna.phy.mapping import Mapper, Demapper  # type: ignore
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder  # type: ignore
        from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder  # type: ignore
        import tensorflow as tf  # type: ignore

        return {
            "BinarySource": BinarySource,
            "AWGN": AWGN,
            "Mapper": Mapper,
            "Demapper": Demapper,
            "LDPC5GEncoder": LDPC5GEncoder,
            "LDPC5GDecoder": LDPC5GDecoder,
            "tf": tf,
            "backend": "sionna.phy",
            "sionna_version": str(getattr(sionna, "__version__", "unknown")),
        }
    except Exception as e:
        errs.append(f"sionna.phy import failed: {e}")

    try:
        import sionna  # type: ignore
        from sionna.utils import BinarySource  # type: ignore
        from sionna.channel import AWGN  # type: ignore
        from sionna.mapping import Mapper, Demapper  # type: ignore
        from sionna.fec.ldpc.encoding import LDPC5GEncoder  # type: ignore
        from sionna.fec.ldpc.decoding import LDPC5GDecoder  # type: ignore
        import tensorflow as tf  # type: ignore

        return {
            "BinarySource": BinarySource,
            "AWGN": AWGN,
            "Mapper": Mapper,
            "Demapper": Demapper,
            "LDPC5GEncoder": LDPC5GEncoder,
            "LDPC5GDecoder": LDPC5GDecoder,
            "tf": tf,
            "backend": "sionna.legacy",
            "sionna_version": str(getattr(sionna, "__version__", "unknown")),
        }
    except Exception as e:
        errs.append(f"sionna legacy import failed: {e}")

    raise RuntimeError("Could not import Sionna NR blocks. " + " | ".join(errs))


def _fer_fallback_awgn(sinr_db: float, modulation_order: int, code_rate: float) -> float:
    """Emergency fallback only.

    This is not publication-grade FER. It exists only to keep light experiments
    from crashing when Sionna is unavailable.
    """
    snr_lin = 10.0 ** (sinr_db / 10.0)
    ebn0 = snr_lin / max(modulation_order * code_rate, 1e-9)
    q = 0.5 * math.erfc(math.sqrt(max(ebn0, 1e-12)))
    fer = 1.0 - (1.0 - q) ** 256
    return float(np.clip(fer, 1e-8, 1.0))


def _make_mapper(Mapper, modulation_order: int):
    for args, kwargs in [
        ((), {"constellation_type": "qam", "num_bits_per_symbol": modulation_order}),
        (("qam", modulation_order), {}),
    ]:
        try:
            if kwargs:
                return Mapper(**kwargs)
            return Mapper(*args)
        except Exception:
            continue
    raise RuntimeError("Could not construct Sionna Mapper")


def _make_demapper(Demapper, modulation_order: int):
    for args, kwargs in [
        (("app",), {"constellation_type": "qam", "num_bits_per_symbol": modulation_order}),
        (("app", "qam", modulation_order), {}),
    ]:
        try:
            if kwargs:
                return Demapper(*args, **kwargs)
            return Demapper(*args)
        except Exception:
            continue
    raise RuntimeError("Could not construct Sionna Demapper")


def _awgn_call(awgn, x, no):
    for payload in ([x, no], (x, no)):
        try:
            return awgn(payload)
        except Exception:
            continue
    try:
        return awgn(x, no)
    except Exception as e:
        raise RuntimeError(f"Could not call AWGN block: {e}") from e


def _demapper_call(demapper, y, no):
    for payload in ([y, no], (y, no)):
        try:
            return demapper(payload)
        except Exception:
            continue
    try:
        return demapper(y, no)
    except Exception as e:
        raise RuntimeError(f"Could not call Demapper: {e}") from e


def _to_hard_bits(x, tf):
    x = tf.cast(x, tf.float32)
    xmin = float(tf.reduce_min(x).numpy())
    xmax = float(tf.reduce_max(x).numpy())
    if xmin < -0.5 or xmax > 1.5:
        return tf.cast(x > 0.0, tf.float32)
    return tf.cast(x > 0.5, tf.float32)


def simulate_5g_nr_fer_curve(
    *,
    sinr_db_points: Sequence[float],
    modulation_order: int,
    code_rate: float,
    k_bits: int = 1024,
    num_frames_per_point: int = 256,
    max_frame_errors: int = 200,
    decoder_iterations: int = 20,
    require_sionna: bool = False,
    allow_fallback: bool = True,
) -> FerCurveResult:
    n_bits = int(math.ceil(k_bits / max(code_rate, 1e-6)))
    n_bits += (-n_bits) % int(modulation_order)

    try:
        blk = _import_sionna_blocks()
        tf = blk["tf"]
        BinarySource = blk["BinarySource"]
        AWGN = blk["AWGN"]
        Mapper = blk["Mapper"]
        Demapper = blk["Demapper"]
        LDPC5GEncoder = blk["LDPC5GEncoder"]
        LDPC5GDecoder = blk["LDPC5GDecoder"]

        encoder = LDPC5GEncoder(k_bits, n_bits)

        decoder = None
        decoder_errs = []
        for kwargs in [
            {"hard_out": True, "num_iter": decoder_iterations},
            {"hard_out": True},
            {"num_iter": decoder_iterations},
            {},
        ]:
            try:
                decoder = LDPC5GDecoder(encoder, **kwargs)
                break
            except Exception as e:
                decoder_errs.append(f"{kwargs}: {e}")
        if decoder is None:
            raise RuntimeError("Could not construct LDPC5GDecoder. " + " | ".join(decoder_errs))

        source = BinarySource()
        mapper = _make_mapper(Mapper, modulation_order=modulation_order)
        demapper = _make_demapper(Demapper, modulation_order=modulation_order)
        awgn = AWGN()

        rows: List[Dict[str, Any]] = []
        bs = int(min(max(num_frames_per_point, 16), 1024))
        for sinr_db in sinr_db_points:
            no = tf.constant(10.0 ** (-float(sinr_db) / 10.0), dtype=tf.float32)
            num_err = 0
            num_tot = 0

            while num_tot < num_frames_per_point and num_err < max_frame_errors:
                cur_bs = min(bs, num_frames_per_point - num_tot)
                u = tf.cast(source([cur_bs, k_bits]), tf.float32)
                c = encoder(u)
                x = mapper(c)
                y = _awgn_call(awgn, x, no)
                llr = _demapper_call(demapper, y, no)
                u_hat = decoder(llr)
                u_hat_bits = _to_hard_bits(u_hat, tf)
                fe = tf.reduce_sum(
                    tf.cast(
                        tf.reduce_any(tf.not_equal(u_hat_bits, u), axis=-1),
                        tf.int32,
                    )
                )
                num_err += int(fe.numpy())
                num_tot += cur_bs

            rows.append(
                {
                    "sinr_db": float(sinr_db),
                    "fer": float(num_err / max(num_tot, 1)),
                    "num_frames": float(num_tot),
                    "num_frame_errors": float(num_err),
                    "used_sionna": True,
                    "backend": str(blk["backend"]),
                    "sionna_version": str(blk["sionna_version"]),
                    "sionna_error": "",
                }
            )

        return FerCurveResult(
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            used_sionna=True,
            backend=str(blk["backend"]),
            sionna_version=str(blk["sionna_version"]),
            error_message="",
            rows=rows,
        )
    except Exception as e:
        err = str(e)
        if require_sionna or not allow_fallback:
            raise RuntimeError(f"Sionna FER failed: {err}") from e

        rows = [
            {
                "sinr_db": float(s),
                "fer": _fer_fallback_awgn(
                    float(s),
                    modulation_order=modulation_order,
                    code_rate=code_rate,
                ),
                "num_frames": float(num_frames_per_point),
                "num_frame_errors": float(np.nan),
                "used_sionna": False,
                "backend": "fallback_awgn",
                "sionna_version": "",
                "sionna_error": err,
            }
            for s in sinr_db_points
        ]
        return FerCurveResult(
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            used_sionna=False,
            backend="fallback_awgn",
            sionna_version="",
            error_message=err,
            rows=rows,
        )


def fer_from_algorithm_summary(
    summary_df: pd.DataFrame,
    *,
    algorithm_col: str = "algorithm",
    sweep_col: str = "sweep_value",
    sinr_col: str = "avg_sinr_db",
    modulation_orders: Sequence[int] = (2, 4),
    code_rates: Sequence[float] = (0.3, 0.5),
    k_bits: int = 1024,
    num_frames_per_point: int = 256,
    max_frame_errors: int = 200,
    decoder_iterations: int = 20,
    require_sionna: bool = False,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for algo, df_algo in summary_df.groupby(algorithm_col):
        sinr_points = list(df_algo[sinr_col].astype(float).tolist())
        sweep_points = list(df_algo[sweep_col].tolist())
        for m in modulation_orders:
            for r in code_rates:
                curve = simulate_5g_nr_fer_curve(
                    sinr_db_points=sinr_points,
                    modulation_order=int(m),
                    code_rate=float(r),
                    k_bits=int(k_bits),
                    num_frames_per_point=int(num_frames_per_point),
                    max_frame_errors=int(max_frame_errors),
                    decoder_iterations=int(decoder_iterations),
                    require_sionna=bool(require_sionna),
                    allow_fallback=bool(allow_fallback),
                )
                for sw, row in zip(sweep_points, curve.rows):
                    rows.append(
                        {
                            algorithm_col: str(algo),
                            sweep_col: sw,
                            "modulation_order": int(m),
                            "code_rate": float(r),
                            "k_bits": int(curve.k_bits),
                            "n_bits": int(curve.n_bits),
                            "fer": float(row["fer"]),
                            "used_sionna": bool(row["used_sionna"]),
                            "backend": str(row["backend"]),
                            "sionna_version": str(row["sionna_version"]),
                            "sionna_error": str(row["sionna_error"]),
                        }
                    )
    return pd.DataFrame(rows)
