"""Network-based statistic for a two-group comparison with covariate adjustment."""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import NbsConfig

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

    @classmethod
    def complete(cls, n_nodes):
        """Every pair of distinct nodes."""
        return cls(np.triu_indices(n_nodes, k=1), n_nodes)

    def values(self, fc, rows=None):
        """Edge-wise values of a symmetric FC stack, shape (len(rows), n_edges)."""
        stack = fc if rows is None else fc[rows]
        return stack[:, self.index[0], self.index[1]].astype(np.float64)

    def nodes_of(self, selection):
        """Node index covered by a set of edge indices."""
        return np.unique(np.concatenate([self.index[0][selection], self.index[1][selection]]))


@dataclass(frozen=True)
class Nbs:
    """Group comparison result; components are edge-index arrays per sign."""

    n: int
    n_hc: int
    n_cud: int
    dof: int
    t_crit: float
    t_values: np.ndarray
    edges: Edges
    selections: dict
    components: dict
    null_max: dict

    def p_fwe(self, sign, size):
        """Family-wise p-value of a component of `size` edges on one side."""
        null = self.null_max[sign]
        return (1 + int(np.sum(null >= size))) / (1 + len(null))


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


def observed_sides(t, edges, t_crit):
    """Suprathreshold edges per sign, with their connected components."""
    sel = {s: np.where(s * t > t_crit)[0] for s in SIGNS}
    return sel, {s: components(sel[s], edges) for s in SIGNS}


def compare(Y, group, covariates, edges, config=NbsConfig(), seed=0):
    """Edge-wise t, suprathreshold components and their permutation null.

    Args:
        Y: (subjects, edges) values; Fisher-z transformed when `config.fisher_z`.
        group: (subjects,) coded 1 for the patient group and 0 for controls.
        covariates: (subjects, terms) nuisance regressors, without the intercept.
        edges: pair index used to build components.
        config: permutation count and edge threshold.
        seed: permutation seed.

    Raises:
        ValueError: If Y has non-finite values, or the design is singular.
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group, dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(f"Y has {int((~np.isfinite(Y)).sum())} non-finite values")
    if config.fisher_z:
        Y = np.arctanh(np.clip(Y, -1 + 1e-7, 1 - 1e-7))
    Z = np.column_stack([np.ones(len(Y)), np.asarray(covariates, dtype=float)])
    model = fit_design(Y, Z)
    t_crit = stats.t.ppf(1 - config.alpha_edge / 2, model.dof)
    t_obs = model.edge_t(group)
    selections, observed = observed_sides(t_obs, edges, t_crit)

    rng = np.random.default_rng(seed)
    null_max = {s: np.empty(config.n_perm, dtype=int) for s in SIGNS}
    for p in range(config.n_perm):
        _, permuted = observed_sides(model.edge_t(rng.permutation(group)), edges, t_crit)
        for s in SIGNS:
            null_max[s][p] = max((c.size for c in permuted[s]), default=0)

    return Nbs(n=len(Y), n_hc=int((group == 0).sum()), n_cud=int((group == 1).sum()),
               dof=model.dof, t_crit=float(t_crit), t_values=t_obs, edges=edges,
               selections=selections, components=observed, null_max=null_max)
