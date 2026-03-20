from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf

from .config import ResolvedConfig


@dataclass(frozen=True)
class TopologyData:
    bs_loc: tf.Tensor
    ut_loc: tf.Tensor
    serving_bs: tf.Tensor
    bs_azimuth_deg: tf.Tensor


@dataclass(frozen=True)
class FixedServiceLocations:
    fs_loc: tf.Tensor


def _hex_sites(num_sites: int, intersite_distance_m: float) -> np.ndarray:
    if num_sites <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    a = float(intersite_distance_m)
    dirs = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
        [-0.5, np.sqrt(3.0) / 2.0],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0],
        [0.5, -np.sqrt(3.0) / 2.0],
    ])
    pts: List[np.ndarray] = [np.array([0.0, 0.0])]
    if num_sites == 1:
        return np.asarray(pts, dtype=np.float32)
    ring = 1
    while len(pts) < num_sites:
        q = ring * dirs[4]
        for d in range(6):
            for _ in range(ring):
                if len(pts) >= num_sites:
                    break
                pts.append(q.copy())
                q = q + dirs[d]
        ring += 1
    return a * np.asarray(pts[:num_sites], dtype=np.float32)


def generate_hexgrid_topology(cfg: ResolvedConfig, batch_size: int = 1) -> TopologyData:
    batch_size = int(batch_size)
    num_sites = int(cfg.raw["pcp"]["num_sites"])
    sectors_per_site = int(cfg.raw["pcp"]["sectors_per_site"])
    u_per_bs = int(cfg.raw["topology"]["num_ut_per_sector"])
    num_bs = int(cfg.derived["num_bs"])
    num_ut = int(cfg.derived["num_ut"])
    intersite = float(cfg.raw["pcp"].get("intersite_distance_m", 500.0))
    min_r = float(cfg.raw["topology"].get("min_ue_distance_m", 20.0))
    cell_r = float(cfg.raw["topology"].get("cell_radius_m", 250.0))

    site_xy = _hex_sites(num_sites, intersite_distance_m=intersite)
    site_xy = np.repeat(site_xy, repeats=sectors_per_site, axis=0)
    site_xy = site_xy[:num_bs]

    if sectors_per_site == 3:
        az = np.tile(np.array([0.0, 120.0, 240.0], dtype=np.float32), reps=num_sites)[:num_bs]
    else:
        az = np.linspace(0.0, 360.0, num_bs, endpoint=False, dtype=np.float32)

    bs = np.repeat(site_xy[None, :, :], repeats=batch_size, axis=0).astype(np.float32)
    ut = np.zeros((batch_size, num_ut, 2), dtype=np.float32)
    serving_bs = np.repeat(np.arange(num_bs, dtype=np.int32), repeats=u_per_bs)

    for s in range(batch_size):
        cursor = 0
        for b in range(num_bs):
            phi0 = np.deg2rad(float(az[b]))
            span = np.pi / max(sectors_per_site, 1)
            angle = np.random.uniform(phi0 - 0.5 * span, phi0 + 0.5 * span, size=u_per_bs)
            r = np.sqrt(np.random.uniform(min_r**2, cell_r**2, size=u_per_bs))
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            ut[s, cursor : cursor + u_per_bs, 0] = bs[s, b, 0] + dx.astype(np.float32)
            ut[s, cursor : cursor + u_per_bs, 1] = bs[s, b, 1] + dy.astype(np.float32)
            cursor += u_per_bs

    return TopologyData(
        bs_loc=tf.convert_to_tensor(bs, dtype=tf.float32),
        ut_loc=tf.convert_to_tensor(ut, dtype=tf.float32),
        serving_bs=tf.convert_to_tensor(serving_bs, dtype=tf.int32),
        bs_azimuth_deg=tf.convert_to_tensor(az, dtype=tf.float32),
    )


def generate_fixed_service_locations(cfg: ResolvedConfig, topo: TopologyData, batch_size: int = 1) -> FixedServiceLocations:
    batch_size = int(batch_size)
    if not bool(cfg.raw.get("fixed_service", {}).get("enabled", False)):
        return FixedServiceLocations(fs_loc=tf.zeros([batch_size, 0, 2], dtype=tf.float32))

    L = int(cfg.raw.get("fixed_service", {}).get("num_receivers", 0))
    bs = topo.bs_loc.numpy()
    out = np.zeros((batch_size, L, 2), dtype=np.float32)
    for s in range(batch_size):
        center = np.mean(bs[s], axis=0)
        radius = float(np.max(np.linalg.norm(bs[s] - center[None, :], axis=-1))) + 0.7 * float(
            cfg.raw["topology"].get("cell_radius_m", 250.0)
        )
        ang = np.random.uniform(0.0, 2.0 * np.pi, size=L)
        r = np.random.uniform(0.9 * radius, 1.4 * radius, size=L)
        out[s, :, 0] = center[0] + r * np.cos(ang)
        out[s, :, 1] = center[1] + r * np.sin(ang)
    return FixedServiceLocations(fs_loc=tf.convert_to_tensor(out, dtype=tf.float32))
