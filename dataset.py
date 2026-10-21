"""Sample selection and loading for the SUDMEX-CONN cocaine-use-disorder study.

Everything that knows about this dataset lives here: the exclusion list, the
covariate names, the archive layout and the output naming.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = frozenset({"sub-154", "sub-129", "sub-136"})
COVARIATES = ("age", "sex", "educ_yr", "mean_fd")
DUMMY = 5        # non-steady-state volumes, as in the paper
STATES = {"integrated": 0, "segregated": 1}


@dataclass(frozen=True)
class Sample:
    """Time series of the analysis sample, dummy volumes already dropped."""

    timeseries: np.ndarray    # (subjects, frames, parcels)
    subjects: list[str]
    parcels: list[str]
    dropped_parcels: int


def read_covariates(root):
    """Covariate table from `<root>/data/covariates.tsv`."""
    return pd.read_csv(Path(root) / "data" / "covariates.tsv", sep="\t", dtype={"sex": str})


def complete_cases(cov, columns=COVARIATES):
    """Rows outside EXCLUDE with every covariate present, plus numeric columns."""
    nums = {c: pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in columns}
    finite = np.isfinite(np.column_stack(list(nums.values()))).all(axis=1)
    return finite & ~cov["subject"].isin(EXCLUDE).to_numpy(), nums


def load_timeseries(root, *, limit=0):
    """Load the analysis sample in covariate order.

    Raises:
        ValueError: If a selected subject is absent from the archive.
    """
    cov = read_covariates(root)
    keep, _ = complete_cases(cov)
    cov = cov[keep].reset_index(drop=True)
    archive = np.load(Path(root) / "data" / "timeseries_all.npz")
    values, names = archive["ts"], list(archive["subjects"])
    index = {name: i for i, name in enumerate(names)}
    missing = [s for s in cov["subject"] if s not in index]
    if missing:
        raise ValueError(f"Subjects missing from timeseries_all.npz: {missing}")
    if limit:
        cov = cov.iloc[:limit]
    ts = values[[index[s] for s in cov["subject"]]][:, DUMMY:, :].astype(np.float64)
    # One all-NaN parcel column collapses the Louvain partition into singletons.
    parcel_ok = ~np.isnan(ts).any(axis=(0, 1))
    return Sample(
        timeseries=ts[:, :, parcel_ok],
        subjects=list(cov["subject"]),
        parcels=[p for p, ok in zip(archive["parcels"], parcel_ok) if ok],
        dropped_parcels=int((~parcel_ok).sum()),
    )


def sample_rows(cov, avail, columns=COVARIATES):
    """Rows with a complete graph and complete covariates, plus their group labels."""
    keep, nums = complete_cases(cov, columns)
    rows = np.where(avail.all(axis=1) & keep)[0]
    return rows, cov["group"].to_numpy()[rows], nums


def load_state_fc(root, state):
    """Mean FC of one dynamic state, realigned to the covariate table."""
    cov = read_covariates(root)
    archive = np.load(Path(root) / "data" / "dfc_states.npz")
    parcels = list(archive["parcels"])
    cent = archive["centroids"][:, STATES[state]]
    # The archive holds only the analysis sample, so realign by subject name.
    position = {s: i for i, s in enumerate(archive["subjects"])}
    fc = np.full((len(cov),) + cent.shape[1:], np.nan)
    hit = cov["subject"].isin(position).to_numpy()
    fc[hit] = cent[[position[s] for s in cov.loc[hit, "subject"]]]
    return fc, np.isfinite(fc[:, 0, 0])[:, None], parcels, cov


def load_static_fc(root):
    """Window-mean FC matrices, for the run without state selection."""
    archive = np.load(Path(root) / "data" / "fc_matrices.npz")
    return archive["fc"], archive["available"], list(archive["parcels"]), read_covariates(root)


def output_suffix(state=None, *, alpha_edge=0.05, fisher_z=True):
    """Output suffix that keeps threshold and state sweeps from overwriting each other.

    The `_noz` tag only applies without a state: state centroids are already Fisher-z.
    """
    threshold = "" if alpha_edge == 0.05 else f"_a{alpha_edge:g}".replace(".", "")
    if state:
        return f"_dfc_{state}" + threshold
    return ("" if fisher_z else "_noz") + threshold
