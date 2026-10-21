"""两组比较的网络基础统计（NBS），含协变量校正。"""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import NbsConfig

SIGNS = (1, -1)          # 正、负分量分开保留


@dataclass(frozen=True)
class Design:
    """对边做偏回归后的残差，以及设计矩阵的帽子矩阵。"""

    Yr: np.ndarray
    ssr: np.ndarray
    hat: np.ndarray
    dof: int

    def edge_t(self, g):
        """给定分组向量 g 时每条边的精确 t 统计量。

        参数:
            g: 长度为被试数的分组向量。

        返回:
            长度为边数的 t 值数组。
        """
        gr = g - self.hat @ g
        gg = gr @ gr
        num = gr @ self.Yr
        sse = np.maximum(self.ssr - num ** 2 / gg, 0.0)
        return num / np.sqrt(np.maximum(sse / self.dof * gg, 1e-300))


@dataclass(frozen=True)
class Edges:
    """上三角边索引与节点数。"""

    index: tuple[np.ndarray, np.ndarray]
    n_nodes: int

    @classmethod
    def complete(cls, n_nodes):
        """覆盖所有互不相同的节点对。

        参数:
            n_nodes: 节点数。

        返回:
            对应的 Edges 记录。
        """
        return cls(np.triu_indices(n_nodes, k=1), n_nodes)

    def values(self, fc, rows=None):
        """取出对称 FC 堆栈中每条边的取值。

        参数:
            fc: 形状为 (被试, 分区, 分区) 的 FC 堆栈。
            rows: 要取的行索引，None 表示全部。

        返回:
            形状为 (行数, 边数) 的数组。
        """
        stack = fc if rows is None else fc[rows]
        return stack[:, self.index[0], self.index[1]].astype(np.float64)

    def nodes_of(self, selection):
        """一组边覆盖的节点索引。

        参数:
            selection: 边索引数组。

        返回:
            去重后的节点索引。
        """
        return np.unique(np.concatenate([self.index[0][selection], self.index[1][selection]]))


@dataclass(frozen=True)
class Nbs:
    """比较结果；分量是按正负号分开的边索引数组。"""

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
        """单侧分量的族错误率 p 值。

        参数:
            sign: 1 或 -1，表示分量所在的一侧。
            size: 分量的边数。

        返回:
            置换分布中不小于该边数的比例。
        """
        null = self.null_max[sign]
        return (1 + int(np.sum(null >= size))) / (1 + len(null))


def fit_design(Y, Z):
    """对边做偏回归，保留分组对比所需的量。

    参数:
        Y: 形状为 (被试, 边数) 的取值。
        Z: 形状为 (被试, 项数) 的设计矩阵，含截距。

    返回:
        Design 记录。

    异常:
        ValueError: 设计矩阵的项数不少于被试数，矩阵会奇异。
    """
    if Z.shape[1] >= len(Y):
        raise ValueError(f"被试数必须多于模型项数：n={len(Y)}，项数={Z.shape[1]}")
    hat = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    Yr = Y - hat @ Y
    return Design(Yr=Yr, ssr=(Yr ** 2).sum(axis=0), hat=hat, dof=len(Y) - Z.shape[1] - 1)


def components(sel, edges):
    """选出边的连通分量，单点分量丢弃。

    参数:
        sel: 边索引数组。
        edges: 边索引与节点数。

    返回:
        分量列表，每个分量是该分量内的边索引数组。
    """
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
    """按正负号分开取超阈边，并各自求连通分量。

    参数:
        t: 长度为边数的 t 值数组。
        edges: 边索引与节点数。
        t_crit: 边阈值。

    返回:
        (每一侧的超阈边索引, 每一侧的分量列表)。
    """
    sel = {s: np.where(s * t > t_crit)[0] for s in SIGNS}
    return sel, {s: components(sel[s], edges) for s in SIGNS}


def compare(Y, group, covariates, edges, config=NbsConfig(), seed=0):
    """逐边 t 统计量、超阈分量及其置换零分布。

    参数:
        Y: 形状为 (被试, 边数) 的取值；config.fisher_z 为真时先做 Fisher-z。
        group: 长度被试数的分组向量，1 为病例组、0 为对照组。
        covariates: 形状为 (被试, 项数) 的协变量，不含截距。
        edges: 用于构建分量的边索引。
        config: 置换次数与边阈值。
        seed: 置换的随机种子。

    返回:
        Nbs 记录。

    异常:
        ValueError: Y 含非有限值、分组编码不是 0/1，或设计矩阵项数不少于被试数。
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group, dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(f"Y 含 {int((~np.isfinite(Y)).sum())} 个非有限值")
    if not set(np.unique(group).tolist()) <= {0.0, 1.0}:
        raise ValueError("分组必须编码为 1（病例）或 0（对照）")
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
