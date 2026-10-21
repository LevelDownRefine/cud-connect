"""Analysis parameters shared by the state and NBS stages."""

from dataclasses import dataclass

WIN_TR = 22               # 44 s window
STEP_TR = 1
DUMMY = 5                 # non-steady-state volumes, as in the paper
SEED = 0
PC_RANGE = (0.0, 1.0)     # participation coefficient is bounded by construction
Z_RANGE = (-3.0, 3.0)     # within-module degree z-score, BCT's usual display range
N_BINS = 40
COVARIATES = ("age", "sex", "educ_yr", "mean_fd")


@dataclass(frozen=True)
class StateConfig:
    """Runnable settings for state estimation."""

    n_bins: int = N_BINS
    kmeans_ninit: int = 300
    cpus: int = 4
    metric: str = "correlation"    # the paper clusters on correlation distance
    seed: int = SEED
    all_parcels: bool = True
    limit: int = 0


@dataclass(frozen=True)
class NbsConfig:
    """Network-based statistic settings; `state=None` uses window-mean FC."""

    state: str | None = None
    n_perm: int = 10000
    alpha_edge: float = 0.05
    fisher_z: bool = True
    all_parcels: bool = False
    covariates: tuple[str, ...] = COVARIATES

    def suffix(self) -> str:
        """Output suffix that keeps threshold sweeps from overwriting each other."""
        threshold = "" if self.alpha_edge == 0.05 else f"_a{self.alpha_edge:g}".replace(".", "")
        if self.state:
            parcels = "_all" if self.all_parcels else ""
            return f"_dfc_{self.state}{parcels}{threshold}"
        return ("" if self.fisher_z else "_noz") + threshold
