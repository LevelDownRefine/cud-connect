"""Estimate integrated and segregated states from parcel time series.

Sliding-window FC -> cartographic profiles -> subject-level k-means (k=2).
State 0: integrated; state 1: segregated. Outputs state labels and mean FC.
Method reference and implementation differences: see README.md.
Run through: python run.py --stage states --cpus 4
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
KMEANS_NINIT = int(sys.argv[3]) if len(sys.argv) > 3 else 300
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0     # 0 = all subjects
CPUS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
ALL_PARCELS = len(sys.argv) > 6 and sys.argv[6] == "all"
SFX = "_all" if ALL_PARCELS else ""

import bct  # noqa: E402

A = os.path.join(ROOT, "analysis")
EXCLUDE = {"sub-154", "sub-129", "sub-136"}
COVS = ["age", "sex", "educ_yr", "mean_fd"]

TR = 2.0                 # seconds
WIN_TR = 22              # 44 s window
STEP_TR = 1
DUMMY = 5                # volumes dropped as non-steady-state, as in the paper
PC_RANGE = (0.0, 1.0)    # participation coefficient is bounded by construction
Z_RANGE = (-3.0, 3.0)    # within-module degree z-score, BCT's usual display range
SEED = 0

cov = pd.read_csv(os.path.join(A, "covariates.tsv"), sep="\t", dtype={"sex": str})
d = np.load(os.path.join(A, "timeseries_all.npz" if ALL_PARCELS else "timeseries.npz"),
            allow_pickle=True)
ts_all, subjects_all, parcels = d["ts"], list(d["subjects"]), list(d["parcels"])

nums = {c: pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in COVS}
cc = np.isfinite(np.column_stack([nums[c] for c in COVS])).all(axis=1)
keep = (~cov["subject"].isin(EXCLUDE).to_numpy()) & cc
cov = cov[keep].reset_index(drop=True)
sample = list(cov["subject"])
idx = [subjects_all.index(s) for s in sample]
if LIMIT:
    sample, idx = sample[:LIMIT], idx[:LIMIT]

ts = ts_all[idx][:, DUMMY:, :].astype(np.float64)         # (n, T-DUMMY, P)
# Louvain needs a complete matrix, and one all-NaN parcel column collapses the
# partition into singletons, so drop every parcel that an included subject is
# missing entirely.  roi_metrics keeps those parcels and leans on NaN-aware
# summaries instead, hence the slightly different parcel count here.
parcel_ok = ~np.isnan(ts).any(axis=(0, 1))
ts = ts[:, :, parcel_ok]
parcels = [p for p, ok in zip(parcels, parcel_ok) if ok]
n, T, P = ts.shape
n_win = T - WIN_TR + 1
print(f"被试 {n}  时间点 {T}（截掉前 {DUMMY}）  分区 {P}"
      f"（{len(parcel_ok)} 中剔除 {int((~parcel_ok).sum())} 个有整条缺失的）"
      f"  窗口 {n_win}（{WIN_TR} TR = {WIN_TR * TR:.0f} s，步长 {STEP_TR} TR）")
print(f"图谱 bins {N_BINS}×{N_BINS}  k-means n_init={KMEANS_NINIT}  进程 {CPUS}")






def profiles_one(subject_ts):
    """Cartographic profiles for one subject: (n_win, N_BINS**2) + the window FCs."""
    fc, prof = [], []
    for w in range(n_win):
        block = subject_ts[w:w + WIN_TR]
        W = np.corrcoef(block, rowvar=False)
        W = np.arctanh(np.clip(W, -1 + 1e-7, 1 - 1e-7))
        np.fill_diagonal(W, 0.0)
        ci, _ = bct.community_louvain(W, B="negative_asym", seed=SEED)
        pcp, _ = bct.participation_coef_sign(W, ci)
        z = bct.module_degree_zscore(W, ci)
        h, _, _ = np.histogram2d(pcp, z, bins=N_BINS, range=[PC_RANGE, Z_RANGE])
        prof.append(h.ravel())
        fc.append(W.astype(np.float32))
    p = np.asarray(prof, float)
    p /= p.sum(axis=1, keepdims=True)
    return p, np.asarray(fc)




def one_subject(subject_ts):
    """Returns (profiles, window FCs, labels, centroids, silhouette, n_states)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    prof, fc = profiles_one(subject_ts)
    km = KMeans(n_clusters=2, n_init=KMEANS_NINIT, random_state=SEED).fit(prof)
    lab = km.labels_
    sil = float(silhouette_score(prof, lab)) if len(set(lab)) > 1 else np.nan
    # align on the mean participation coefficient read off the histogram margin:
    # the state with the higher mean PC is "integrated"
    pc_center = (np.arange(N_BINS) + 0.5) * (PC_RANGE[1] - PC_RANGE[0]) / N_BINS
    mean_pc = [(prof[lab == k].reshape(-1, N_BINS, N_BINS).sum(axis=2)
                * pc_center).sum(axis=1).mean() if (lab == k).any() else np.nan
               for k in (0, 1)]
    integ = int(mean_pc[1] > mean_pc[0])        # False when state 1 is empty
    lab = np.where(lab == integ, 0, 1)          # 0 = integrated, 1 = segregated
    nan_cent = np.full((P, P), np.nan)
    centroids = np.stack([fc[lab == k].mean(axis=0) if (lab == k).any() else nan_cent
                          for k in (0, 1)])
    return prof, fc, lab, centroids, sil, int(np.unique(lab).size)







def worker(arg):
    i, subject_ts = arg
    _, _, lab, cent, sil, ns = one_subject(subject_ts)
    return i, lab, cent, sil, ns


if __name__ == "__main__":
    t_start = time.time()
    with Pool(min(CPUS, n)) as pool:
        res = pool.map(worker, [(i, ts[i]) for i in range(n)], chunksize=1)
    labels = np.empty((n, n_win), dtype=np.uint8)
    centroids = np.empty((n, 2, P, P), dtype=np.float32)
    sil = np.empty(n)
    n_states = np.empty(n, dtype=int)
    for i, lab, cent, score, ns in res:
        labels[i], centroids[i], sil[i], n_states[i] = lab, cent, score, ns
    degen = [sample[i] for i in np.where(n_states < 2)[0]]
    if degen or not np.isfinite(centroids).all():
        raise ValueError(f"Invalid state centroids; single-state subjects: {degen}")
    np.savez_compressed(os.path.join(A, f"dfc_states{SFX}.npz"),
                        subjects=np.array(sample), labels=labels,
                        centroids=centroids, parcels=np.array(parcels))
    pd.DataFrame({"subject": sample, "silhouette": sil, "n_states": n_states}).to_csv(
        os.path.join(A, f"dfc_state_quality{SFX}.tsv"), sep="\t", index=False)
    meta = {"n": n, "timepoints": int(T), "windows": int(n_win), "parcels": int(P),
            "parcels_dropped_for_nan": int((~parcel_ok).sum()),
            "parcel_source": "all-256" if ALL_PARCELS else "cortical-only",
            "win_tr": WIN_TR, "step_tr": STEP_TR, "dummy_volumes": DUMMY,
            "louvain_B": "negative_asym", "louvain_repeats": 1,
            "n_bins": N_BINS, "kmeans_ninit": KMEANS_NINIT,
            "kmeans_metric": "Euclidean", "seed": SEED,
            "n_subjects_single_state": len(degen),
            "scope": "state estimation and state-specific connectivity only",
            "elapsed_s": round(time.time() - t_start, 1)}
    with open(os.path.join(A, f"dfc_meta{SFX}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved dfc_states{SFX}.npz, state quality and metadata; {time.time()-t_start:.0f}s")
