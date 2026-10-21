"""Sample selection and loading of the parcel time series and covariates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COVARIATES, DUMMY, StateConfig

EXCLUDE = frozenset({"sub-154", "sub-129", "sub-136"})


@dataclass(frozen=True)
class TimeSeries:
    """Time series of the analysis sample, dummy volumes already dropped."""

    timeseries: np.ndarray    # (subjects, frames, parcels)
    subjects: list[str]
    parcels: list[str]
    dropped_parcels: int


def read_covariates(root):
    """Covariate table from `<root>/analysis/covariates.tsv`."""
    return pd.read_csv(Path(root) / "analysis" / "covariates.tsv", sep="\t",
                       dtype={"sex": str})


def complete_cases(cov, columns=COVARIATES):
    """Rows outside EXCLUDE with every covariate present, plus numeric columns."""
    nums = {c: pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in columns}
    finite = np.isfinite(np.column_stack(list(nums.values()))).all(axis=1)
    return finite & ~cov["subject"].isin(EXCLUDE).to_numpy(), nums


def load_timeseries(root, config=StateConfig()):
    """Load the analysis sample in covariate order.

    Raises:
        ValueError: If a selected subject is absent from the archive.
    """
    cov = read_covariates(root)
    keep, _ = complete_cases(cov)
    cov = cov[keep].reset_index(drop=True)
    source = "timeseries_all.npz" if config.all_parcels else "timeseries.npz"
    archive = np.load(Path(root) / "analysis" / source, allow_pickle=True)
    values, names = archive["ts"], list(archive["subjects"])
    index = {name: i for i, name in enumerate(names)}
    missing = [s for s in cov["subject"] if s not in index]
    if missing:
        raise ValueError(f"Subjects missing from {source}: {missing}")
    if config.limit:
        cov = cov.iloc[:config.limit]
    ts = values[[index[s] for s in cov["subject"]]][:, DUMMY:, :].astype(np.float64)
    # One all-NaN parcel column collapses the Louvain partition into singletons.
    parcel_ok = ~np.isnan(ts).any(axis=(0, 1))
    return TimeSeries(
        timeseries=ts[:, :, parcel_ok],
        subjects=list(cov["subject"]),
        parcels=[p for p, ok in zip(archive["parcels"], parcel_ok) if ok],
        dropped_parcels=int((~parcel_ok).sum()),
    )
