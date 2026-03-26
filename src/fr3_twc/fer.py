from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import math

import numpy as np
import pandas as pd


_MIN_SIONNA_EFFECTIVE_CODE_RATE = 1.0 / 5.0
_SIONNA_RATE_TOL = 1.0e-12


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


def _aligned_n_bits(k_bits: int, code_rate: float, modulation_order: int) -> int:
    n_bits = int(math.ceil(int(k_bits) / max(float(code_rate), 1.0e-6)))
    n_bits += (-n_bits) % int(modulation_order)
    return int(n_bits)


def _effective_code_rate(k_bits: int, *, n_bits: int) -> float:
    return float(int(k_bits) / max(int(n_bits), 1))


def _invalid_sionna_fer_grid_messages(
    *,
    modulation_orders: Sequence[int],
    code_rates: Sequence[float],
    k_bits: int,
) -> list[str]:
    msgs: list[str] = []
    for m in modulation_orders:
        m_int = int(m)
        if m_int <= 0:
            msgs.append(f"invalid modulation order {m!r}")
            continue
        for r in code_rates:
            r_float = float(r)
            if not math.isfinite(r_float) or r_float <= 0.0:
                msgs.append(f"invalid code rate {r!r}")
                continue
            n_bits = _aligned_n_bits(k_bits=int(k_bits), code_rate=r_float, modulation_order=m_int)
            eff_rate = _effective_code_rate(int(k_bits), n_bits=n_bits)
            if eff_rate + _SIONNA_RATE_TOL < _MIN_SIONNA_EFFECTIVE_CODE_RATE:
                msgs.append(
                    "unsupported Sionna FER grid entry: "
                    f"modulation_order={m_int}, requested_code_rate={r_float:.6f}, "
                    f"k_bits={int(k_bits)}, aligned_n_bits={n_bits}, "
                    f"effective_code_rate={eff_rate:.6f} < 0.200000"
                )
    return msgs


def validate_sionna_fer_grid(
    *,
    modulation_orders: Sequence[int],
    code_rates: Sequence[float],
    k_bits: int,
) -> None:
    msgs = _invalid_sionna_fer_grid_messages(
        modulation_orders=modulation_orders,
        code_rates=code_rates,
        k_bits=k_bits,
    )
    if msgs:
        raise ValueError("Invalid FER grid for strict Sionna evaluation: " + " | ".join(msgs))


def _import_sionna_blocks() -> Dict[str, Any]:
    errs: list[str] = []

    try:
        import sionna  # type: ignore
        import tensorflow as tf  # type: ignore
        from sionna.phy.channel import AWGN  # type: ignore
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder  # type: ignore
        from sionna.phy.mapping import BinarySource, Demapper, Mapper  # type: ignore

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
    except Exception as e:  # pragma: no cover - depends on environment
        errs.append(f"sionna.phy public API import failed: {e}")

    try:
        import sionna  # type: ignore
        import tensorflow as tf  # type: ignore
        from sionna.phy.channel.awgn import AWGN  # type: ignore
        from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder  # type: ignore
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder  # type: ignore
        from sionna.phy.mapping import BinarySource, Demapper, Mapper  # type: ignore

        return {
            "BinarySource": BinarySource,
            "AWGN": AWGN,
            "Mapper": Mapper,
            "Demapper": Demapper,
            "LDPC5GEncoder": LDPC5GEncoder,
            "LDPC5GDecoder": LDPC5GDecoder,
            "tf": tf,
            "backend": "sionna.phy.submodule",
            "sionna_version": str(getattr(sionna, "__version__", "unknown")),
        }
    except Exception as e:  # pragma: no cover - depends on environment
        errs.append(f"sionna.phy submodule import failed: {e}")

    raise RuntimeError("Could not import Sionna FER blocks. " + " | ".join(errs))


def _fer_fallback_awgn(sinr_db: float, modulation_order: int, code_rate: float) -> float:
    """Emergency fallback only.

    This is not publication-grade FER. It exists only to keep light experiments
    from crashing when Sionna is unavailable.
    """
    snr_lin = 10.0 ** (float(sinr_db) / 10.0)
    ebn0 = snr_lin / max(int(modulation_order) * float(code_rate), 1.0e-9)
    q = 0.5 * math.erfc(math.sqrt(max(ebn0, 1.0e-12)))
    fer = 1.0 - (1.0 - q) ** 256
    return float(np.clip(fer, 1.0e-8, 1.0))


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


def _fallback_curve_result(
    *,
    sinr_db_points: Sequence[float],
    modulation_order: int,
    code_rate: float,
    k_bits: int,
    n_bits: int,
    num_frames_per_point: int,
    error_message: str,
) -> FerCurveResult:
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
            "sionna_error": str(error_message),
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
        error_message=str(error_message),
        rows=rows,
    )


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
    n_bits = _aligned_n_bits(
        k_bits=int(k_bits),
        code_rate=float(code_rate),
        modulation_order=int(modulation_order),
    )
    eff_rate = _effective_code_rate(int(k_bits), n_bits=n_bits)
    invalid_msgs = _invalid_sionna_fer_grid_messages(
        modulation_orders=[int(modulation_order)],
        code_rates=[float(code_rate)],
        k_bits=int(k_bits),
    )
    if invalid_msgs:
        err = invalid_msgs[0]
        if bool(require_sionna) or not bool(allow_fallback):
            raise RuntimeError(err)
        return _fallback_curve_result(
            sinr_db_points=sinr_db_points,
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            num_frames_per_point=int(num_frames_per_point),
            error_message=err,
        )

    try:
        blk = _import_sionna_blocks()
        tf = blk["tf"]
        BinarySource = blk["BinarySource"]
        AWGN = blk["AWGN"]
        Mapper = blk["Mapper"]
        Demapper = blk["Demapper"]
        LDPC5GEncoder = blk["LDPC5GEncoder"]
        LDPC5GDecoder = blk["LDPC5GDecoder"]

        encoder = LDPC5GEncoder(
            int(k_bits),
            int(n_bits),
            num_bits_per_symbol=int(modulation_order),
        )

        decoder = None
        decoder_errs = []
        for kwargs in [
            {"hard_out": True, "return_infobits": True, "num_iter": decoder_iterations},
            {"hard_out": True, "return_infobits": True},
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
        mapper = _make_mapper(Mapper, modulation_order=int(modulation_order))
        demapper = _make_demapper(Demapper, modulation_order=int(modulation_order))
        awgn = AWGN()

        rows: List[Dict[str, Any]] = []
        bs = int(min(max(int(num_frames_per_point), 16), 1024))
        for sinr_db in sinr_db_points:
            no = tf.constant(10.0 ** (-float(sinr_db) / 10.0), dtype=tf.float32)
            num_err = 0
            num_tot = 0

            while num_tot < int(num_frames_per_point) and num_err < int(max_frame_errors):
                cur_bs = min(bs, int(num_frames_per_point) - num_tot)
                u = tf.cast(source([cur_bs, int(k_bits)]), tf.float32)
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
                    "effective_code_rate": float(eff_rate),
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
        if bool(require_sionna) or not bool(allow_fallback):
            raise RuntimeError(f"Sionna FER failed: {err}") from e

        return _fallback_curve_result(
            sinr_db_points=sinr_db_points,
            modulation_order=int(modulation_order),
            code_rate=float(code_rate),
            k_bits=int(k_bits),
            n_bits=int(n_bits),
            num_frames_per_point=int(num_frames_per_point),
            error_message=err,
        )


def _extract_numeric_column(df_algo: pd.DataFrame, col: str) -> list[float] | None:
    if col not in df_algo.columns:
        return None
    vals = pd.to_numeric(df_algo[col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return None
    return list(vals.astype(float))


def _rate_to_sinr(df_algo: pd.DataFrame) -> list[float] | None:
    if "avg_user_rate_bps_per_hz" not in df_algo.columns:
        return None
    avg_rate = pd.to_numeric(
        df_algo["avg_user_rate_bps_per_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(avg_rate).any():
        return None
    sinr_lin = np.maximum(np.power(2.0, avg_rate) - 1.0, 1.0e-9)
    return list((10.0 * np.log10(sinr_lin)).astype(float))


def _infer_fer_input_sinr_db(
    df_algo: pd.DataFrame,
    *,
    requested_col: str,
    fallback_cols: Sequence[str] = ("avg_sinr_db", "p05_sinr_db", "avg_user_rate_bps_per_hz"),
) -> tuple[list[float], str]:
    """Choose the FER-driving SINR metric.

    The previous repo would automatically replace ``avg_sinr_db`` by a
    rate-inverted SINR whenever the two looked inconsistent. In the current
    outputs that produced trivial FER curves (mostly all-zero). Here we use the
    requested metric directly if it exists, and only fall back when it is
    missing.
    """
    candidates = [str(requested_col)] + [
        str(c) for c in fallback_cols if str(c) != str(requested_col)
    ]
    for col in candidates:
        if col == "avg_user_rate_bps_per_hz":
            vals = _rate_to_sinr(df_algo)
            if vals is not None:
                return vals, "rate_inversion"
            continue

        vals = _extract_numeric_column(df_algo, col)
        if vals is not None:
            return vals, str(col)

    raise KeyError(f"Could not infer FER SINR input from columns: {list(df_algo.columns)}")


def fer_from_algorithm_summary(
    summary_df: pd.DataFrame,
    *,
    algorithm_col: str = "algorithm",
    sweep_col: str = "sweep_value",
    sinr_col: str = "p50_sinr_db",
    fallback_sinr_cols: Sequence[str] = ("avg_sinr_db", "p05_sinr_db", "avg_user_rate_bps_per_hz"),
    modulation_orders: Sequence[int] = (2, 4),
    code_rates: Sequence[float] = (0.3, 0.5),
    k_bits: int = 1024,
    num_frames_per_point: int = 256,
    max_frame_errors: int = 200,
    decoder_iterations: int = 20,
    require_sionna: bool = False,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    if bool(require_sionna) or not bool(allow_fallback):
        validate_sionna_fer_grid(
            modulation_orders=modulation_orders,
            code_rates=code_rates,
            k_bits=int(k_bits),
        )

    rows: List[Dict[str, Any]] = []
    for algo, df_algo in summary_df.groupby(algorithm_col):
        sinr_points, sinr_source = _infer_fer_input_sinr_db(
            df_algo,
            requested_col=sinr_col,
            fallback_cols=fallback_sinr_cols,
        )
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
                            "effective_code_rate": float(_effective_code_rate(curve.k_bits, n_bits=curve.n_bits)),
                            "fer": float(row["fer"]),
                            "fer_input_sinr_db": float(row["sinr_db"]),
                            "sinr_source": str(sinr_source),
                            "used_sionna": bool(row["used_sionna"]),
                            "backend": str(row["backend"]),
                            "sionna_version": str(row["sionna_version"]),
                            "sionna_error": str(row["sionna_error"]),
                        }
                    )
    return pd.DataFrame(rows)
