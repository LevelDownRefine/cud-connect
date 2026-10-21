"""Cartographic profiles and integrated/segregated state estimation."""

from dataclasses import dataclass
from multiprocessing import Pool

import bct
import numpy as np

from .config import N_BINS, PC_RANGE, SEED, STEP_TR, WIN_TR, Z_RANGE, StateConfig


@dataclass(frozen=True)
class States:
    """Per-subject state labels and the mean FC of each state."""

    labels: np.ndarray        # (subjects, windows), 0 = integrated
    centroids: np.ndarray     # (subjects, 2, parcels, parcels), Fisher-z
    silhouette: np.ndarray    # (subjects,)
    counts: np.ndarray        # (subjects,) distinct states found


def window_fc(subject_ts):
    """Fisher-z correlation of every full window, shape (windows, parcels, parcels)."""
    subject_ts = np.asarray(subject_ts, dtype=float)
    if subject_ts.ndim != 2 or subject_ts.shape[0] < WIN_TR:
        raise ValueError("Time series must be frames x parcels with at least one full window")
    windows = []
    for start in range(0, subject_ts.shape[0] - WIN_TR + 1, STEP_TR):
        matrix = np.corrcoef(subject_ts[start:start + WIN_TR], rowvar=False)
        matrix = np.arctanh(np.clip(matrix, -1 + 1e-7, 1 - 1e-7))
        np.fill_diagonal(matrix, 0.0)
        windows.append(matrix)
    return np.asarray(windows)


def cartographic(fc, n_bins=N_BINS, seed=SEED):
    """Normalized participation-by-degree profile of each window, one row per window.

    Trick around BCT's seeding: community_louvain builds a fresh RandomState from
    `seed` on every call, so handing every window the same seed makes all windows
    walk the same random sequence. This reproduces the earlier runs, but the Louvain
    perturbations are then correlated across windows instead of independent.
    """
    rows = []
    for matrix in fc:
        communities, _ = bct.community_louvain(matrix, B="negative_asym", seed=seed)
        participation, _ = bct.participation_coef_sign(matrix, communities)
        degree = bct.module_degree_zscore(matrix, communities)
        histogram, _, _ = np.histogram2d(
            participation, degree, bins=n_bins, range=[PC_RANGE, Z_RANGE],
        )
        rows.append(histogram.ravel())
    profiles = np.asarray(rows, float)
    totals = profiles.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        raise ValueError("A window has no nodes inside the cartographic profile ranges")
    return profiles / totals


def _embedding(profiles, metric):
    """Map profiles to the unit sphere so Euclidean distance ranks like correlation.

    sklearn's KMeans has no correlation metric. Centring each row and scaling it to
    unit norm makes squared Euclidean distance equal 2 - 2r, so the assignment step
    ranks exactly like correlation distance. Centroids stay plain means in that
    space rather than spherical means, so this approximates correlation k-means.
    """
    if metric != "correlation":
        return profiles
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)


def cluster_states(profiles, config=StateConfig()):
    """K-means into two states; returns labels, silhouette and the distinct count."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(profiles) < 2:
        raise ValueError("State estimation requires at least two windows")
    x = _embedding(profiles, config.metric)
    labels = KMeans(n_clusters=2, n_init=config.kmeans_ninit,
                    random_state=config.seed).fit_predict(x)
    count = int(np.unique(labels).size)
    silhouette = (float(silhouette_score(x, labels))
                  if 1 < count < len(labels) else np.nan)
    return labels, silhouette, count


def align_states(labels, profiles, fc, n_bins=N_BINS):
    """Relabel so 0 = integrated, then average the window FC of each state."""
    centers = (PC_RANGE[0] + (np.arange(n_bins) + .5)
               * (PC_RANGE[1] - PC_RANGE[0]) / n_bins)
    mean_pc = [
        (profiles[labels == state].reshape(-1, n_bins, n_bins).sum(axis=2)
         * centers).sum(axis=1).mean() if (labels == state).any() else np.nan
        for state in (0, 1)
    ]
    integrated = int(mean_pc[1] > mean_pc[0])
    labels = np.where(labels == integrated, 0, 1)
    empty_centroid = np.full(fc.shape[1:], np.nan)
    centroids = np.stack([fc[labels == state].mean(axis=0)
                          if (labels == state).any() else empty_centroid
                          for state in (0, 1)])
    return labels, centroids


def one_subject(subject_ts, config=StateConfig()):
    """Return profiles, FCs, labels, centroids, silhouette and state count."""
    fc = window_fc(subject_ts)
    profiles = cartographic(fc, config.n_bins, config.seed)
    labels, silhouette, count = cluster_states(profiles, config)
    labels, centroids = align_states(labels, profiles, fc, config.n_bins)
    return profiles, fc, labels, centroids, silhouette, count


def _worker(job):
    index, subject_ts, config = job
    _, _, labels, centroids, silhouette, count = one_subject(subject_ts, config)
    return index, labels, centroids, silhouette, count


def estimate(timeseries, config=StateConfig(), workers=1):
    """Estimate states for every subject and stack the outputs.

    Raises:
        ValueError: If no subject is left, or if any centroid is not finite.
    """
    n, frames = timeseries.shape[:2]
    if n == 0:
        raise ValueError("No subjects to estimate")
    if workers < 1:
        raise ValueError("workers must be positive")
    windows = len(range(0, frames - WIN_TR + 1, STEP_TR))
    jobs = [(index, subject_ts, config) for index, subject_ts in enumerate(timeseries)]
    if workers == 1:
        results = map(_worker, jobs)
    else:
        with Pool(min(workers, n)) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
    parcels = timeseries.shape[2]
    labels = np.empty((n, windows), dtype=np.uint8)
    centroids = np.empty((n, 2, parcels, parcels), dtype=np.float32)
    silhouette = np.empty(n)
    counts = np.empty(n, dtype=int)
    for index, lab, cent, score, count in results:
        labels[index], centroids[index] = lab, cent
        silhouette[index], counts[index] = score, count
    if not np.isfinite(centroids).all():
        raise ValueError("Invalid state centroids; check for single-state subjects")
    return States(labels=labels, centroids=centroids, silhouette=silhouette, counts=counts)
