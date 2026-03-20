"""Core baseline simulation package for FR3 coexistence studies.

This lightweight replacement restores the missing ``fr3_sim`` package that the
TWC extension depends on. It provides:
- config parsing and derived quantities
- reproducible seeding helpers
- reference multi-cell topology generation
- UE and FS channel/statistics generation
- MMSE combiner / SINR / MSE processing
"""

from .config import ResolvedConfig
from .channel import FsStats, generate_fs_stats, generate_ue_channels
from .topology import FixedServiceLocations, TopologyData, generate_fixed_service_locations, generate_hexgrid_topology
from .processing import MmseOutput, mmse_combiners_and_mse
from .seeding import set_global_seed

__all__ = [
    "ResolvedConfig",
    "FsStats",
    "generate_fs_stats",
    "generate_ue_channels",
    "FixedServiceLocations",
    "TopologyData",
    "generate_fixed_service_locations",
    "generate_hexgrid_topology",
    "MmseOutput",
    "mmse_combiners_and_mse",
    "set_global_seed",
]
