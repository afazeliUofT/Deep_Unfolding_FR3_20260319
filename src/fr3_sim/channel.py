from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import ResolvedConfig
from .topology import FixedServiceLocations, TopologyData


_FS_CSV_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class FsStats:
    bar_beta: tf.Tensor
    epsilon: tf.Tensor
    delta: tf.Tensor
    i_max_watt: tf.Tensor
    correlation: str = "identity"
    a_bs_fs: Optional[tf.Tensor] = None


def _precision_dtypes(cfg: ResolvedConfig) -> Tuple[tf.DType, tf.DType]:
    prec = str(cfg.derived.get("tf_precision", "single")).lower()
    if prec.startswith("single"):
        return tf.float32, tf.complex64
    return tf.float64, tf.complex128


def _complex_normal_np(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    z = (
        np.random.standard_normal(shape)
        + 1j * np.random.standard_normal(shape)
    ) / np.sqrt(2.0)
    return z.astype(dtype)


def _path_gain_from_distance(
    d_m: np.ndarray,
    fc_ghz: float,
    shadow_std_db: float = 6.0,
) -> np.ndarray:
    d = np.maximum(d_m, 1.0)
    pl_db = 32.4 + 20.0 * np.log10(max(float(fc_ghz), 1e-3)) + 31.9 * np.log10(d)
    if shadow_std_db > 0.0:
        pl_db = pl_db + np.random.normal(
            loc=0.0,
            scale=float(shadow_std_db),
            size=d.shape,
        )
    beta = 10.0 ** (-pl_db / 10.0)
    return beta.astype(np.float32)


def _repo_root() -> Path:
    import os

    if os.environ.get("FR3_REPO_ROOT"):
        return Path(os.environ["FR3_REPO_ROOT"])
    return Path.cwd()


def _find_fs_csvs(repo_root: Path) -> list[Path]:
    data_root = repo_root / "data"
    if not data_root.exists():
        return []

    csvs = sorted(p for p in data_root.rglob("*.csv") if p.is_file())

    def _priority(p: Path) -> tuple[int, int, str]:
        full = str(p).lower()
        name = p.name.lower()
        is_ised = int(not ("ised" in full or "sms" in full or "fixed" in name))
        is_root = int(p.parent != data_root)
        return (is_ised, is_root, full)

    return sorted(csvs, key=_priority)


def _optional_fs_specs(cfg: ResolvedConfig, L: int) -> Tuple[np.ndarray, np.ndarray]:
    repo_root = _repo_root()
    csvs = _find_fs_csvs(repo_root)

    tone_centers = np.linspace(0.0, 1.0, max(L, 1), endpoint=False, dtype=np.float32)
    i_max_dbm = np.full(
        (L,),
        float(cfg.raw.get("fixed_service", {}).get("i_max_dbm", -110.0)),
        dtype=np.float32,
    )

    if not csvs:
        return tone_centers, i_max_dbm

    cache_key = f"{csvs[0]}::{L}"
    if cache_key in _FS_CSV_CACHE:
        return _FS_CSV_CACHE[cache_key]

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csvs[0], low_memory=False)
        if len(df) > 0:
            rows = df.iloc[np.arange(L) % len(df)]
            numeric = {str(c).lower(): c for c in df.columns}

            for key in [
                "i_max_dbm",
                "imax_dbm",
                "threshold_dbm",
                "protection_dbm",
                "rx_threshold_dbm",
            ]:
                if key in numeric:
                    vals = rows[numeric[key]].astype(float).to_numpy(dtype=np.float32)
                    i_max_dbm = vals
                    break

            for key in ["center_fraction", "center_norm", "fc_frac"]:
                if key in numeric:
                    vals = np.mod(
                        rows[numeric[key]].astype(float).to_numpy(dtype=np.float32),
                        1.0,
                    )
                    tone_centers = vals
                    break
    except Exception:
        pass

    _FS_CSV_CACHE[cache_key] = (tone_centers, i_max_dbm)
    return _FS_CSV_CACHE[cache_key]


def _dbm_to_watt(x_dbm: np.ndarray) -> np.ndarray:
    x_dbm = np.asarray(x_dbm, dtype=np.float32)
    # Correct conversion: P[W] = 10^((P[dBm]-30)/10)
    return np.power(10.0, (x_dbm - 30.0) / 10.0, dtype=np.float32)


def generate_ue_channels(
    cfg: ResolvedConfig,
    topo: TopologyData,
    batch_size: int = 1,
) -> tf.Tensor:
    real_dtype, complex_dtype = _precision_dtypes(cfg)
    S = int(batch_size)
    B = int(cfg.derived["num_bs"])
    U = int(cfg.derived["num_ut"])
    M = int(cfg.raw["channel_model"]["num_bs_ant"])
    Nr = int(cfg.raw["channel_model"]["num_ut_ant"])
    fc_ghz = float(
        cfg.derived.get(
            "carrier_frequency_ghz",
            cfg.raw["channel_model"].get("carrier_frequency_ghz", 7.0),
        )
    )
    shadow_std = float(cfg.raw["channel_model"].get("shadow_fading_std_db", 6.0))
    bs = topo.bs_loc.numpy()
    ut = topo.ut_loc.numpy()
    diff = bs[:, :, None, :] - ut[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    beta = _path_gain_from_distance(d, fc_ghz=fc_ghz, shadow_std_db=shadow_std)
    H = _complex_normal_np(
        (S, B, U, Nr, M),
        np.complex64 if complex_dtype == tf.complex64 else np.complex128,
    )
    H = H * np.sqrt(beta[:, :, :, None, None].astype(H.real.dtype))
    return tf.convert_to_tensor(H, dtype=complex_dtype)


def _steering_ula(angles_rad: np.ndarray, num_ant: int, dtype: np.dtype) -> np.ndarray:
    n = np.arange(int(num_ant), dtype=np.float32)
    phase = np.pi * np.sin(angles_rad)[..., None] * n[None, ...]
    a = np.exp(1j * phase) / np.sqrt(max(int(num_ant), 1))
    return a.astype(dtype)


def generate_fs_stats(
    cfg: ResolvedConfig,
    topo: TopologyData,
    fs_loc: FixedServiceLocations,
    batch_size: int = 1,
) -> FsStats:
    real_dtype, complex_dtype = _precision_dtypes(cfg)
    S = int(batch_size)
    B = int(cfg.derived["num_bs"])
    T = int(cfg.derived["num_re_sim"])
    M = int(cfg.raw["channel_model"]["num_bs_ant"])
    L = int(cfg.raw.get("fixed_service", {}).get("num_receivers", 0))
    fc_ghz = float(
        cfg.derived.get(
            "carrier_frequency_ghz",
            cfg.raw["channel_model"].get("carrier_frequency_ghz", 7.0),
        )
    )
    shadow_std = float(cfg.raw["channel_model"].get("shadow_fading_std_db", 6.0))

    bs = topo.bs_loc.numpy()
    fs = fs_loc.fs_loc.numpy()
    diff = bs[:, :, None, :] - fs[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    bar_beta = _path_gain_from_distance(
        d,
        fc_ghz=fc_ghz,
        shadow_std_db=max(0.5 * shadow_std, 0.0),
    )

    tone_centers, i_max_dbm = _optional_fs_specs(cfg, L)
    tone_axis = np.linspace(0.0, 1.0, T, endpoint=False, dtype=np.float32)[:, None]
    width = max(0.18, 1.0 / max(T, 2))
    epsilon = np.exp(
        -0.5 * ((tone_axis - tone_centers[None, :]) / width) ** 2
    ).astype(np.float32)
    epsilon /= np.maximum(np.max(epsilon, axis=0, keepdims=True), 1e-6)
    epsilon = np.clip(0.15 + 0.85 * epsilon, 1e-3, 1.0)

    delta = np.ones((B, T), dtype=np.float32)

    # This threshold enters both training and evaluation. It must be correct.
    i_max_watt = _dbm_to_watt(i_max_dbm).astype(np.float32)

    correlation = str(cfg.raw.get("antenna", {}).get("correlation", "identity"))
    a_bs_fs = None
    if correlation == "steering_rank1":
        ang = np.arctan2(
            fs[:, None, :, 1] - bs[:, :, None, 1],
            fs[:, None, :, 0] - bs[:, :, None, 0],
        ).astype(np.float32)
        a_bs_fs = _steering_ula(
            ang,
            num_ant=M,
            dtype=np.complex64 if complex_dtype == tf.complex64 else np.complex128,
        )

    return FsStats(
        bar_beta=tf.convert_to_tensor(bar_beta, dtype=real_dtype),
        epsilon=tf.convert_to_tensor(epsilon, dtype=real_dtype),
        delta=tf.convert_to_tensor(delta, dtype=real_dtype),
        i_max_watt=tf.convert_to_tensor(i_max_watt, dtype=real_dtype),
        correlation=correlation,
        a_bs_fs=None if a_bs_fs is None else tf.convert_to_tensor(a_bs_fs, dtype=complex_dtype),
    )
