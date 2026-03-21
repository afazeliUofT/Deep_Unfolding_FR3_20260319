from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Tuple

import copy
import math
import yaml


@dataclass(frozen=True)
class ResolvedConfig:
    raw: Dict[str, Any]
    derived: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"raw": copy.deepcopy(self.raw), "derived": copy.deepcopy(self.derived)}


_DEFAULTS: Dict[str, Any] = {
    "reproducibility": {
        "seed": 1,
        "deterministic_tf": True,
        "precision": "single",
    },
    "system_model": {
        "num_re_total": 273,
        "num_re_sim": 4,
        "subcarrier_spacing_hz": 30000.0,
        "bs_total_tx_power_dbm": 49.0,
        "simulate_full_band": False,
    },
    "noise": {
        "noise_figure_db": 7.0,
        "extra_noise_db": 0.0,
    },
    "topology": {
        "num_ut_per_sector": 4,
        "cell_radius_m": 250.0,
        "min_ue_distance_m": 20.0,
    },
    "pcp": {
        "num_sites": 7,
        "sectors_per_site": 3,
        "intersite_distance_m": 500.0,
        "background_sites_fraction": 0.15,
        "ue_cluster_radius_m": 100.0,
        "hotspot_jitter_sigma_m": 350.0,
        "bs_cluster_sigma_m": 900.0,
    },
    "channel_model": {
        "carrier_frequency_ghz": 7.0,
        "num_bs_ant": 16,
        "num_ut_ant": 2,
        "shadow_fading_std_db": 6.0,
        "rician_k_db": 0.0,
    },
    "fixed_service": {
        "enabled": True,
        "num_receivers": 8,
        "i_max_dbm": -110.0,
        "in_target_db": -10.0,
        "sweep_override_i_max": "none",
    },
    "antenna": {
        "correlation": "steering_rank1",
    },
    "receiver": {
        "name": "wmmse",
        "wmmse": {
            "num_iterations": 20,
            "dual_step_mu": 0.02,
            "dual_step_lambda": 0.10,
            "damping_w": 0.90,
            "ridge_regularization": 1.0e-9,
            "convergence_tol": 1.0e-6,
            "aggressive_fs_nulling_reg": 1.0e-6,
        },
    },
    "experiment": {
        "batch_size": 1,
        "num_batches": 4,
        "freeze_topology": True,
        "tf_compile": False,
    },
    "sweep": {
        "enabled": True,
        "variable": "snr_db",
        "values": [0.0],
    },
    "output": {
        "results_root": "results",
        "experiment_name": "baseline_wmmse",
        "overwrite": False,
    },
    "plotting": {
        "enabled": True,
        "format": "png",
        "dpi": 220,
    },
}


def _deep_fill_defaults(cfg: MutableMapping[str, Any], defaults: Dict[str, Any]) -> None:
    for k, v in defaults.items():
        if isinstance(v, dict):
            cur = cfg.get(k)
            if not isinstance(cur, MutableMapping):
                cfg[k] = copy.deepcopy(v)
            else:
                _deep_fill_defaults(cur, v)
        elif k not in cfg:
            cfg[k] = copy.deepcopy(v)


def _parse_override(override: str) -> Tuple[List[str], Any]:
    if "=" not in str(override):
        raise ValueError(f"Override must be KEY=VALUE, got: {override}")
    lhs, rhs = str(override).split("=", 1)
    keys = [k.strip() for k in lhs.split(".") if k.strip()]
    if not keys:
        raise ValueError(f"Invalid override key path: {override}")
    value = yaml.safe_load(rhs)
    return keys, value


def _normalize_legacy_blocks(cfg: Dict[str, Any]) -> None:
    """Accept the current repo YAML layout even if some blocks were nested wrongly.

    The pushed ``twc_base.yaml`` places ``pcp`` under ``topology`` and ``antenna``
    under ``fixed_service``. This helper lifts them to the expected root keys so
    the config still resolves correctly.
    """
    topo = cfg.get("topology")
    if isinstance(topo, MutableMapping):
        pcp = topo.pop("pcp", None)
        if isinstance(pcp, MutableMapping):
            root_pcp = cfg.get("pcp") if isinstance(cfg.get("pcp"), MutableMapping) else {}
            root_pcp = copy.deepcopy(root_pcp)
            root_pcp.update(copy.deepcopy(dict(pcp)))
            cfg["pcp"] = root_pcp

    fs = cfg.get("fixed_service")
    if isinstance(fs, MutableMapping):
        ant = fs.pop("antenna", None)
        if isinstance(ant, MutableMapping):
            root_ant = cfg.get("antenna") if isinstance(cfg.get("antenna"), MutableMapping) else {}
            root_ant = copy.deepcopy(root_ant)
            root_ant.update(copy.deepcopy(dict(ant)))
            cfg["antenna"] = root_ant


def _validate_minimum(cfg: Dict[str, Any]) -> None:
    _normalize_legacy_blocks(cfg)
    _deep_fill_defaults(cfg, _DEFAULTS)

    num_sites = int(cfg["pcp"]["num_sites"])
    sectors_per_site = int(cfg["pcp"]["sectors_per_site"])
    u_per_bs = int(cfg["topology"]["num_ut_per_sector"])
    num_re_total = int(cfg["system_model"]["num_re_total"])
    num_re_sim = int(cfg["system_model"].get("num_re_sim", num_re_total))
    if num_sites <= 0 or sectors_per_site <= 0 or u_per_bs <= 0:
        raise ValueError("num_sites, sectors_per_site, and num_ut_per_sector must be positive")
    if num_re_total <= 0 or num_re_sim <= 0:
        raise ValueError("num_re_total and num_re_sim must be positive")
    if num_re_sim > num_re_total:
        cfg["system_model"]["num_re_sim"] = int(num_re_total)


def _dbm_to_watt(dbm: float) -> float:
    # Correct conversion: P[W] = 10^((P[dBm]-30)/10).
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def _derive(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _validate_minimum(cfg)

    num_sites = int(cfg["pcp"]["num_sites"])
    sectors_per_site = int(cfg["pcp"]["sectors_per_site"])
    num_bs = num_sites * sectors_per_site
    u_per_bs = int(cfg["topology"]["num_ut_per_sector"])
    num_ut = num_bs * u_per_bs

    num_re_total = int(cfg["system_model"]["num_re_total"])
    num_re_sim = int(cfg["system_model"].get("num_re_sim", num_re_total))
    re_scaling = float(num_re_total) / float(max(num_re_sim, 1))

    p_dbm = float(cfg["system_model"]["bs_total_tx_power_dbm"])
    bs_total_tx_power_watt = _dbm_to_watt(p_dbm)

    precision = str(cfg["reproducibility"].get("precision", "single"))
    scs_hz = float(cfg["system_model"]["subcarrier_spacing_hz"])
    noise_fig_db = float(cfg["noise"].get("noise_figure_db", 7.0))
    extra_noise_db = float(cfg["noise"].get("extra_noise_db", 0.0))
    noise_dbm = -174.0 + 10.0 * math.log10(max(scs_hz, 1.0)) + noise_fig_db + extra_noise_db
    ue_noise_re_watt = _dbm_to_watt(noise_dbm)

    fc_ghz = float(cfg["channel_model"].get("carrier_frequency_ghz", 7.0))
    derived: Dict[str, Any] = {
        "num_bs": int(num_bs),
        "num_ut": int(num_ut),
        "u_per_bs": int(u_per_bs),
        "num_re_total": int(num_re_total),
        "num_re_sim": int(num_re_sim),
        "re_scaling": float(re_scaling),
        "bs_total_tx_power_watt": float(bs_total_tx_power_watt),
        "ue_noise_re_watt": float(ue_noise_re_watt),
        "tf_precision": precision,
        "carrier_frequency_ghz": float(fc_ghz),
        "carrier_frequency_hz": float(fc_ghz * 1e9),
        "num_fs": int(cfg["fixed_service"].get("num_receivers", 0)),
    }
    return derived
