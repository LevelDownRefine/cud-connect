"""Network-based statistic for group differences with covariate adjustment."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import COVARIATES, NbsConfig
from .data import complete_cases, read_covariates

SIGNS = (1, -1)          # positive and negative components are kept apart


@dataclass(frozen=True)
class Design:
    """Partially residualized edges plus the hat matrix of the design."""

    Yr: np.ndarray
    ssr: np.ndarray
    hat: np.ndarray
    dof: int

    def edge_t(self, g):
        """Exact group t-statistic per edge for group vector g."""
        gr = g - self.hat @ g
        gg = gr @ gr
        num = gr @ self.Yr
        sse = np.maximum(self.ssr - num ** 2 / gg, 0.0)
        return num / np.sqrt(np.maximum(sse / self.dof * gg, 1e-300))


@dataclass(frozen=True)
class Edges:
    """Upper-triangle index of the parcel pairs and the node count."""

    index: tuple[np.ndarray, np.ndarray]
    n_nodes: int


def fit_design(Y, Z):
    """Residualize edges on Z and keep what the group contrast needs.

    Raises:
        ValueError: If the design has as many terms as subjects; it would be singular.
    """
    if Z.shape[1] >= len(Y):
        raise ValueError(f"Need more subjects than model terms: n={len(Y)}, terms={Z.shape[1]}")
    hat = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    Yr = Y - hat @ Y
    return Design(Yr=Yr, ssr=(Yr ** 2).sum(axis=0), hat=hat, dof=len(Y) - Z.shape[1] - 1)


def load_edges(root, config=NbsConfig()):
    """Return (fc, avail, parcels, cov) for the requested source."""
    root = Path(root)
    cov = read_covariates(root)
    if config.state:
        archive = np.load(root / "data"
                          / f"dfc_states{'_all' if config.all_parcels else ''}.npz",
                          allow_pickle=True)
        parcels = list(archive["parcels"])
        cent = archive["centroids"][:, {"integrated": 0, "segregated": 1}[config.state]]
        # The archive holds only the analysis sample, so realign by subject name.
        pos = {s: i for i, s in enumerate(archive["subjects"])}
        fc = np.full((len(cov),) + cent.shape[1:], np.nan)
        hit = cov["subject"].isin(pos).to_numpy()
        fc[hit] = cent[[pos[s] for s in cov.loc[hit, "subject"]]]
        avail = np.isfinite(fc[:, 0, 0])[:, None]
    else:
        archive = np.load(root / "data" / "fc_matrices.npz", allow_pickle=True)
        fc, avail, parcels = archive["fc"], archive["available"], list(archive["parcels"])
    return fc, avail, parcels, cov


def components(sel, edges):
    """Connected components of the selected edges; singletons dropped."""
    if sel.size == 0:
        return []
    r, c = edges.index[0][sel], edges.index[1][sel]
    ncomp, labels = connected_components(
        coo_matrix((np.ones(sel.size), (r, c)), shape=(edges.n_nodes, edges.n_nodes)),
        directed=False)
    out = []
    for k in range(ncomp):
        nodes = np.where(labels == k)[0]
        if nodes.size >= 2:
            out.append(sel[np.isin(r, nodes) & np.isin(c, nodes)])
    return out


def analysis_rows(cov, avail, covariates):
    """Rows with a complete graph and complete covariates, plus their group labels."""
    keep, nums = complete_cases(cov, covariates)
    rows = np.where(avail.all(axis=1) & keep)[0]
    return rows, cov["group"].to_numpy()[rows], nums


def observed_sides(t, edges, t_crit):
    """Suprathreshold edges per sign, with their connected components."""
    sel = {s: np.where(s * t > t_crit)[0] for s in SIGNS}
    return sel, {s: components(sel[s], edges) for s in SIGNS}


def run_nbs(root=".", config=NbsConfig()):
    """Run the group comparison and write summary, edge table and null distribution."""
    out = Path(root) / "data"
    fc, avail, parcels, cov = load_edges(root, config)
    networks = pd.read_csv(out / "parcels_cortex.tsv", sep="\t")
    net = dict(zip(networks["name"], networks["network_label"]))

    rows, groups, nums = analysis_rows(cov, avail, config.covariates)
    edges = Edges(np.triu_indices(len(parcels), k=1), len(parcels))
    print(f"完整图 + 协变量完整: n={len(rows)}"
          f"（HC {int((groups == 'HC').sum())} / CUD {int((groups == 'CUD').sum())}）"
          f"  节点 {len(parcels)}  边 {len(edges.index[0])}")
    print(f"边阈值 alpha={config.alpha_edge}   Fisher-z={'on' if config.fisher_z else 'off'}"
          f"   n_perm={config.n_perm}   协变量={config.covariates}")

    Y = fc[rows[:, None], edges.index[0], edges.index[1]].astype(np.float64)
    if config.fisher_z:
        Y = np.arctanh(np.clip(Y, -1 + 1e-7, 1 - 1e-7))
    n = len(rows)
    print(f"Y {Y.shape}  NaN {int((~np.isfinite(Y)).sum())}")

    Z = np.column_stack([np.ones(n)] + [nums[c][rows] for c in config.covariates])
    model = fit_design(Y, Z)
    print(f"残差化完成  dof={model.dof}")

    t_crit = stats.t.ppf(1 - config.alpha_edge / 2, model.dof)
    print(f"t_crit={t_crit:.4f}")

    group = (groups == "CUD").astype(float)
    start = time.time()
    t_obs = model.edge_t(group)
    sel, found = observed_sides(t_obs, edges, t_crit)
    for s in SIGNS:
        big = max((c.size for c in found[s]), default=0)
        print(f"观测[{'+' if s > 0 else '-'}]: 超阈边 {sel[s].size:4d}"
              f"  分量 {len(found[s]):3d}  最大分量 {big:4d} 条边")
    print(f"  ({time.time() - start:.1f}s)")

    rng = np.random.default_rng(0)
    null_max = {s: np.empty(config.n_perm, dtype=int) for s in SIGNS}
    start = time.time()
    for p in range(config.n_perm):
        _, permuted = observed_sides(model.edge_t(rng.permutation(group)), edges, t_crit)
        for s in SIGNS:
            null_max[s][p] = max((c.size for c in permuted[s]), default=0)
        if (p + 1) % 2000 == 0:
            print(f"   置换 {p + 1}/{config.n_perm}  ({time.time() - start:.0f}s)", flush=True)
    print(f"置换完成 {time.time() - start:.0f}s")
    for s in SIGNS:
        tag = "+" if s > 0 else "-"
        print(f"  零分布[{tag}]: 中位 {np.median(null_max[s]):.0f}"
              f"  p95 {np.percentile(null_max[s], 95):.0f}  max {null_max[s].max():.0f}")

    report = {"n": n, "n_hc": int((group == 0).sum()), "n_cud": int((group == 1).sum()),
              "n_perm": config.n_perm, "alpha_edge": config.alpha_edge,
              "fisher_z": bool(config.fisher_z), "t_crit": float(t_crit),
              "covariates": list(config.covariates), "nodes": len(parcels),
              "edges": int(len(edges.index[0])), "components": []}

    print(f"\n{'#':>3} {'边数':>6} {'节点':>5} {'sum t':>8} {'p(FWE)':>8}  网络构成")
    edges_out = []
    for s in SIGNS:
        tag = "+" if s > 0 else "-"
        for k, c in enumerate(sorted(found[s], key=lambda x: -x.size)):
            nodes = np.unique(np.concatenate([edges.index[0][c], edges.index[1][c]]))
            p_fwe = (1 + np.sum(null_max[s] >= c.size)) / (1 + config.n_perm)
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
                i, j = edges.index[0][e], edges.index[1][e]
                edges_out.append((parcels[i], parcels[j],
                                  net.get(parcels[i], "subcortical"),
                                  net.get(parcels[j], "subcortical"), t_obs[e], tag))

    suffix = config.suffix()
    if edges_out:
        pd.DataFrame(edges_out, columns=["node_i", "node_j", "network_i", "network_j",
                                         "t", "sign"]) \
            .sort_values("t", key=abs, ascending=False) \
            .to_csv(out / f"baseline_nbs_edges{suffix}.tsv", sep="\t", index=False)

    significant = [c for c in report["components"] if c["p_fwe"] < 0.05]
    print(f"\nFWE p<0.05 的分量: {len(significant)}"
          + (f"  最大 {significant[0]['sign']}{significant[0]['n_edges']} 条边"
             f" / {significant[0]['n_nodes']} 节点" if significant else ""))

    (out / f"baseline_nbs_summary{suffix}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.save(out / f"baseline_nbs_null_maxsize{suffix}.npy",
            np.stack([null_max[s] for s in SIGNS]))
    print(f"已写出 baseline_nbs_summary{suffix}.json / baseline_nbs_edges{suffix}.tsv "
          f"/ baseline_nbs_null_maxsize{suffix}.npy")
    return report


def main(argv=None):
    """Compatibility CLI for the original nbs.py positional arguments."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_perm", nargs="?", default=10000, type=int)
    parser.add_argument("alpha_edge", nargs="?", default=0.05, type=float)
    parser.add_argument("covariates", nargs="?", default=",".join(COVARIATES))
    parser.add_argument("z", nargs="?", default="z",
                        help='"noz" keeps the values as they are')
    parser.add_argument("state", nargs="?", choices=["integrated", "segregated"])
    parser.add_argument("parcels", nargs="?", choices=["all"])
    args = parser.parse_args(argv)
    run_nbs(args.root, NbsConfig(
        state=args.state, n_perm=args.n_perm, alpha_edge=args.alpha_edge,
        fisher_z=args.z.lower() not in ("0", "no", "noz", "false"),
        all_parcels=args.parcels == "all",
        covariates=tuple(args.covariates.split(","))))

