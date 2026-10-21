"""Analysis parameters shared by the state and NBS stages."""

from dataclasses import dataclass

WIN_TR = 22               # 44 s window
STEP_TR = 1
SEED = 0
PC_RANGE = (0.0, 1.0)     # participation coefficient is bounded by construction
Z_RANGE = (-3.0, 3.0)     # within-module degree z-score, BCT's usual display range
N_BINS = 40


@dataclass(frozen=True)
class StateConfig:
    """Settings for the sliding-window state estimation."""

    n_bins: int = N_BINS
    kmeans_ninit: int = 300
    metric: str = "correlation"    # the paper clusters on correlation distance
    seed: int = SEED


@dataclass(frozen=True)
class NbsConfig:
    """Settings for the network-based statistic."""

    n_perm: int = 10000
    alpha_edge: float = 0.05
    fisher_z: bool = True
