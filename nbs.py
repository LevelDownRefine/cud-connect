"""NBS group comparison with age, sex, education and mean-FD covariates.

Compute edgewise group t-statistics, threshold each sign separately and compare
component edge counts with the permutation distribution of the largest component.
Permute group labels and re-residualize each permutation against covariates.
State centroids are already Fisher-z values: use noz to skip another transform.
Run through: python run.py --stage nbs
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
ALPHA_EDGE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
COVS = (sys.argv[4].split(",") if len(sys.argv) > 4
        else ["age", "sex", "educ_yr", "mean_fd"])
FISHER_Z = len(sys.argv) <= 5 or sys.argv[5].lower() not in ("0", "no", "noz", "false")
STATE = sys.argv[6] if len(sys.argv) > 6 else None
ALLP = len(sys.argv) > 7 and sys.argv[7] == "all"
# every output carries the setting, so a threshold sweep does not overwrite itself
SFX = ("" if FISHER_Z else "_noz") + ("" if ALPHA_EDGE == 0.05
                                      else f"_a{ALPHA_EDGE:g}".replace(".", ""))
if STATE:
    SFX = (f"_dfc_{STATE}" + ("_all" if ALLP else "")
           + ("" if ALPHA_EDGE == 0.05 else f"_a{ALPHA_EDGE:g}".replace(".", "")))

A = os.path.join(ROOT, "analysis")
EXCLUDE = {"sub-154", "sub-129", "sub-136"}
SIGNS = (1, -1)          # positive and negative components are kept apart

cov = pd.read_csv(os.path.join(A, "covariates.tsv"), sep="\t", dtype={"sex": str})
if STATE:
    # dfc_states.npz holds only the analysis sample, so realign by subject name
    ds = np.load(os.path.join(A, f"dfc_states{'_all' if ALLP else ''}.npz"),
                 allow_pickle=True)
    parcels = list(ds["parcels"])
    cent = ds["centroids"][:, {"integrated": 0, "segregated": 1}[STATE]]
    pos = {s: i for i, s in enumerate(ds["subjects"])}
    fc = np.full((len(cov),) + cent.shape[1:], np.nan)
    hit = cov["subject"].isin(pos).to_numpy()
    fc[hit] = cent[[pos[s] for s in cov.loc[hit, "subject"]]]
    avail = np.isfinite(fc[:, 0, 0])[:, None]
else:
    d = np.load(os.path.join(A, "fc_matrices.npz"), allow_pickle=True)
    fc, avail, parcels = d["fc"], d["available"], list(d["parcels"])
lab = pd.read_csv(os.path.join(A, "parcels_cortex.tsv"), sep="\t")
net = dict(zip(lab["name"], lab["network_label"]))

nums = {c: pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in COVS}
cc = np.isfinite(np.column_stack([nums[c] for c in COVS])).all(axis=1)
keep = avail.all(axis=1) & cc & ~cov["subject"].isin(EXCLUDE).to_numpy()
rows = np.where(keep)[0]
g_all = cov["group"].to_numpy()
print(f"完整图 + 协变量完整: n={len(rows)}"
      f"（HC {int((g_all[rows] == 'HC').sum())} / CUD {int((g_all[rows] == 'CUD').sum())}）"
      f"  节点 {len(parcels)}  边 {len(parcels) * (len(parcels) - 1) // 2}")
print(f"边阈值 alpha={ALPHA_EDGE}   Fisher-z={'on' if FISHER_Z else 'off'}"
      f"   n_perm={N_PERM}   协变量={COVS}")

iu = np.triu_indices(len(parcels), k=1)
Y = fc[rows[:, None], iu[0], iu[1]].astype(np.float64)   # (n_subjects, n_edges)
if FISHER_Z:
    Y = np.arctanh(np.clip(Y, -1 + 1e-7, 1 - 1e-7))
n = len(rows)
print(f"Y {Y.shape}  NaN {int((~np.isfinite(Y)).sum())}")

Z = np.column_stack([np.ones(n)] + [nums[c][rows] for c in COVS])
dof = n - Z.shape[1] - 1
HZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
Yr = Y - HZ @ Y
SSr = (Yr ** 2).sum(axis=0)
print(f"残差化完成  dof={dof}")

t_crit = stats.t.ppf(1 - ALPHA_EDGE / 2, dof)
print(f"t_crit={t_crit:.4f}")


def edge_t(g):
    """Exact group t-statistic per edge for group vector g."""
    gr = g - HZ @ g
    gg = gr @ gr
    num = gr @ Yr
    sse = np.maximum(SSr - num ** 2 / gg, 0.0)
    return num / np.sqrt(np.maximum(sse / dof * gg, 1e-300))


def components(t, sign):
    """Connected components of the suprathreshold edges on one side."""
    sel = np.where(sign * t > t_crit)[0]
    if sel.size == 0:
        return [], sel
    r, c = iu[0][sel], iu[1][sel]
    ncomp, labels = connected_components(
        coo_matrix((np.ones(sel.size), (r, c)), shape=(len(parcels), len(parcels))),
        directed=False)
    out = []
    for k in range(ncomp):
        nodes = np.where(labels == k)[0]
        if nodes.size < 2:
            continue
        out.append(sel[np.isin(r, nodes) & np.isin(c, nodes)])
    return out, sel


group = (g_all[rows] == "CUD").astype(float)
t0 = time.time()
t_obs = edge_t(group)
comp_obs = {s: components(t_obs, s) for s in SIGNS}
for s in SIGNS:
    n_edges = sum(c.size for c in comp_obs[s][0])
    big = max((c.size for c in comp_obs[s][0]), default=0)
    print(f"观测[{'+' if s > 0 else '-'}]: 超阈边 {comp_obs[s][1].size:4d}"
          f"  分量 {len(comp_obs[s][0]):3d}  最大分量 {big:4d} 条边")
print(f"  ({time.time() - t0:.1f}s)")

rng = np.random.default_rng(0)
null_max = {s: np.empty(N_PERM, dtype=int) for s in SIGNS}
t1 = time.time()
for p in range(N_PERM):
    tp = edge_t(rng.permutation(group))
    for s in SIGNS:
        null_max[s][p] = max((c.size for c in components(tp, s)[0]), default=0)
    if (p + 1) % 2000 == 0:
        print(f"   置换 {p + 1}/{N_PERM}  ({time.time() - t1:.0f}s)", flush=True)
print(f"置换完成 {time.time() - t1:.0f}s")
for s in SIGNS:
    tag = "+" if s > 0 else "-"
    print(f"  零分布[{tag}]: 中位 {np.median(null_max[s]):.0f}"
          f"  p95 {np.percentile(null_max[s], 95):.0f}  max {null_max[s].max():.0f}")

report = {"n": n, "n_hc": int((group == 0).sum()), "n_cud": int((group == 1).sum()),
          "n_perm": N_PERM, "alpha_edge": ALPHA_EDGE, "fisher_z": bool(FISHER_Z),
          "t_crit": float(t_crit), "covariates": COVS,
          "nodes": len(parcels), "edges": int(len(iu[0])), "components": []}

print(f"\n{'#':>3} {'边数':>6} {'节点':>5} {'sum t':>8} {'p(FWE)':>8}  网络构成")
rows_out = []
for s in SIGNS:
    tag = "+" if s > 0 else "-"
    for k, c in enumerate(sorted(comp_obs[s][0], key=lambda x: -x.size)):
        nodes = np.unique(np.concatenate([iu[0][c], iu[1][c]]))
        p_fwe = (1 + np.sum(null_max[s] >= c.size)) / (1 + N_PERM)
        nets = pd.Series([net.get(parcels[i], "subcortical") for i in nodes]).value_counts()
        print(f"{tag}{k + 1:>2} {c.size:>6} {nodes.size:>5} {t_obs[c].sum():>+8.1f} "
              f"{p_fwe:>8.4f}  " + " ".join(f"{a}:{v}" for a, v in nets.items()))
        report["components"].append({
            "sign": tag, "n_edges": int(c.size), "n_nodes": int(nodes.size),
            "sum_t": float(t_obs[c].sum()), "p_fwe": float(p_fwe),
            "networks": {a: int(v) for a, v in nets.items()},
            "nodes": [parcels[i] for i in nodes],
        })
        for e in c:
            rows_out.append((parcels[iu[0][e]], parcels[iu[1][e]],
                             net.get(parcels[iu[0][e]], "subcortical"),
                             net.get(parcels[iu[1][e]], "subcortical"),
                             t_obs[e], tag))

if rows_out:
    pd.DataFrame(rows_out, columns=["node_i", "node_j", "network_i", "network_j",
                                    "t", "sign"]) \
        .sort_values("t", key=abs, ascending=False) \
        .to_csv(os.path.join(A, f"baseline_nbs_edges{SFX}.tsv"), sep="\t", index=False)

sig = [c for c in report["components"] if c["p_fwe"] < 0.05]
print(f"\nFWE p<0.05 的分量: {len(sig)}"
      + (f"  最大 {sig[0]['sign']}{sig[0]['n_edges']} 条边 / {sig[0]['n_nodes']} 节点"
         if sig else ""))

with open(os.path.join(A, f"baseline_nbs_summary{SFX}.json"), "w") as fh:
    json.dump(report, fh, indent=2)
np.save(os.path.join(A, f"baseline_nbs_null_maxsize{SFX}.npy"),
        np.stack([null_max[s] for s in SIGNS]))
print(f"已写出 baseline_nbs_summary{SFX}.json / baseline_nbs_edges{SFX}.tsv "
      f"/ baseline_nbs_null_maxsize{SFX}.npy")
