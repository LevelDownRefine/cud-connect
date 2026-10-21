"""Cartographic profiles, state estimation and state-output orchestration."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import bct
import numpy as np
import pandas as pd

from .config import DUMMY, N_BINS, PC_RANGE, SEED, STEP_TR, WIN_TR, Z_RANGE, StateConfig
from .data import load_timeseries


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


def cartographic(fc, n_bins=N_BINS):
    """Normalized participation-by-degree profile of each window, one row per window."""
    rows = []
    for matrix in fc:
        communities, _ = bct.community_louvain(matrix, B="negative_asym", seed=SEED)
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


def cluster_states(profiles, config=StateConfig()):
    """K-means into two states; returns labels, silhouette and the distinct count."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(profiles) < 2:
        raise ValueError("State estimation requires at least two windows")
    labels = KMeans(n_clusters=2, n_init=config.kmeans_ninit,
                    random_state=SEED).fit_predict(profiles)
    count = int(np.unique(labels).size)
    silhouette = (float(silhouette_score(profiles, labels))
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
    profiles = cartographic(fc, config.n_bins)
    labels, silhouette, count = cluster_states(profiles, config)
    labels, centroids = align_states(labels, profiles, fc, config.n_bins)
    return profiles, fc, labels, centroids, silhouette, count


def _worker(job):
    index, subject_ts, config = job
    _, _, labels, centroids, silhouette, count = one_subject(subject_ts, config)
    return index, labels, centroids, silhouette, count


def run_states(root, config=StateConfig()):
    """Load time series, estimate each subject's states and save the outputs."""
    if config.cpus < 1:
        raise ValueError("cpus must be positive")
    start = time.time()
    data = load_timeseries(root, config)
    n, frames, parcels = data.timeseries.shape
    if n == 0:
        raise ValueError("No subjects remain after sample selection")
    windows = len(range(0, frames - WIN_TR + 1, STEP_TR))
    print(f"Subjects {n}; frames {frames}; parcels {parcels}; windows {windows}", flush=True)
    jobs = [(index, subject_ts, config) for index, subject_ts in enumerate(data.timeseries)]
    if config.cpus == 1:
        results = map(_worker, jobs)
    else:
        with Pool(min(config.cpus, n)) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
    labels = np.empty((n, windows), dtype=np.uint8)
    centroids = np.empty((n, 2, parcels, parcels), dtype=np.float32)
    silhouette = np.empty(n)
    state_count = np.empty(n, dtype=int)
    for index, lab, cent, score, count in results:
        labels[index], centroids[index] = lab, cent
        silhouette[index], state_count[index] = score, count
    degenerate = [data.subjects[i] for i in np.flatnonzero(state_count < 2)]
    if degenerate or not np.isfinite(centroids).all():
        raise ValueError(f"Invalid state centroids; single-state subjects: {degenerate}")
    out = Path(root) / "data"
    suffix = "_all" if config.all_parcels else ""
    output = out / f"dfc_states{suffix}.npz"
    np.savez_compressed(output, subjects=np.array(data.subjects), labels=labels,
                        centroids=centroids, parcels=np.array(data.parcels))
    pd.DataFrame({"subject": data.subjects, "silhouette": silhouette,
                  "n_states": state_count}).to_csv(
        out / f"dfc_state_quality{suffix}.tsv", sep="\t", index=False)
    meta = {
        "n": n, "timepoints": frames, "windows": windows, "parcels": parcels,
        "parcels_dropped_for_nan": data.dropped_parcels,
        "parcel_source": "all-256" if config.all_parcels else "cortical-only",
        "win_tr": WIN_TR, "step_tr": STEP_TR, "dummy_volumes": DUMMY,
        "louvain_B": "negative_asym", "louvain_repeats": 1,
        "n_bins": config.n_bins, "kmeans_ninit": config.kmeans_ninit,
        "kmeans_metric": "Euclidean", "seed": SEED,
        "n_subjects_single_state": len(degenerate),
        "scope": "state estimation and state-specific connectivity only",
        "elapsed_s": round(time.time() - start, 1),
    }
    (out / f"dfc_meta{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {output.name}, state quality and metadata; {time.time()-start:.0f}s")
    return output


def main(argv=None):
    """Compatibility CLI for the original states.py positional arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_bins", nargs="?", default=N_BINS, type=int)
    parser.add_argument("n_init", nargs="?", default=300, type=int)
    parser.add_argument("limit", nargs="?", default=0, type=int)
    parser.add_argument("cpus", nargs="?", default=8, type=int)
    parser.add_argument("parcels", nargs="?", choices=["all"])
    args = parser.parse_args(argv)
    run_states(args.root, StateConfig(
        n_bins=args.n_bins, kmeans_ninit=args.n_init, cpus=args.cpus,
        all_parcels=args.parcels == "all", limit=args.limit))


if __name__ == "__main__":
    main()
