from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

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
    rows: List[Dict[str, float]]


def _import_sionna_blocks():
    errs = []
    try:
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
        }
    except Exception as e:
        errs.append(str(e))

    try:
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
        }
    except Exception as e:
        errs.append(str(e))

    raise RuntimeError("Could not import Sionna NR blocks. Errors: " + " | ".join(errs))


def _fer_fallback_awgn(sinr_db: float, modulation_order: int, code_rate: float) -> float:
    """Simple emergency fallback used only if Sionna cannot be imported.

    This is **not** a replacement for the Sionna link simulation. It just keeps
    the pipeline from crashing and clearly flags ``used_sionna=False``.
    """
    snr_lin = 10.0 ** (sinr_db / 10.0)
    ebn0 = snr_lin / max(modulation_order * code_rate, 1e-9)
    q = 0.5 * math.erfc(math.sqrt(max(ebn0, 1e-12)))
    fer = 1.0 - (1.0 - q) ** 256
    return float(np.clip(fer, 1e-8, 1.0))


def _make_mapper(Mapper, modulation_order: int):
    for args, kwargs in [
        ((), {"constellation_type": "qam", "num_bits_per_symbol": modulation_order}),
        ((("qam"), modulation_order), {}),
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
        raise RuntimeError(f"Could not call AWGN block: {e}")


def _demapper_call(demapper, y, no):
    for payload in ([y, no], (y, no)):
        try:
            return demapper(payload)
        except Exception:
            continue
    try:
        return demapper(y, no)
    except Exception as e:
        raise RuntimeError(f"Could not call Demapper: {e}")


def simulate_5g_nr_fer_curve(
    *,
    sinr_db_points: Sequence[float],
    modulation_order: int,
    code_rate: float,
    k_bits: int = 1024,
    num_frames_per_point: int = 256,
    max_frame_errors: int = 200,
    decoder_iterations: int = 20,
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
        try:
            decoder = LDPC5GDecoder(encoder, hard_out=True, num_iter=decoder_iterations)
        except Exception:
            try:
                decoder = LDPC5GDecoder(encoder, hard_out=True)
            except Exception:
                decoder = LDPC5GDecoder(encoder)

        source = BinarySource()
        mapper = _make_mapper(Mapper, modulation_order=modulation_order)
        demapper = _make_demapper(Demapper, modulation_order=modulation_order)
        awgn = AWGN()

        rows: List[Dict[str, float]] = []
        bs = int(min(max(num_frames_per_point, 32), 1024))
        for sinr_db in sinr_db_points:
            no = 10.0 ** (-float(sinr_db) / 10.0)
            num_err = 0
            num_tot = 0
            while num_tot < num_frames_per_point and num_err < max_frame_errors:
                cur_bs = min(bs, num_frames_per_point - num_tot)
                u = tf.cast(source([cur_bs, k_bits]), tf.float32)
                c = encoder(u)
                x = mapper(c)
                y = _awgn_call(awgn, x, tf.cast(no, tf.float32))
                llr = _demapper_call(demapper, y, tf.cast(no, tf.float32))
                u_hat = decoder(llr)
                fe = tf.reduce_sum(tf.cast(tf.reduce_any(tf.not_equal(tf.cast(u_hat, tf.float32), u), axis=-1), tf.int32))
                num_err += int(fe.numpy())
                num_tot += cur_bs
            rows.append(
                {
                    "sinr_db": float(sinr_db),
                    "fer": float(num_err / max(num_tot, 1)),
                    "num_frames": float(num_tot),
                    "num_frame_errors": float(num_err),
                    "used_sionna": 1.0,
                }
            )
        return FerCurveResult(
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            used_sionna=True,
            rows=rows,
        )
    except Exception:
        rows = [
            {
                "sinr_db": float(s),
                "fer": _fer_fallback_awgn(float(s), modulation_order=modulation_order, code_rate=code_rate),
                "num_frames": float(num_frames_per_point),
                "num_frame_errors": float(np.nan),
                "used_sionna": 0.0,
            }
            for s in sinr_db_points
        ]
        return FerCurveResult(
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            used_sionna=False,
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
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
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
                            "used_sionna": bool(curve.used_sionna),
                        }
                    )
    return pd.DataFrame(rows)
