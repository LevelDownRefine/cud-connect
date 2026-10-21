"""地形图剖面与整合/分离状态估计。"""

from dataclasses import dataclass
from multiprocessing import Pool

import bct
import numpy as np

from .config import N_BINS, PC_RANGE, SEED, STEP_TR, WIN_TR, Z_RANGE, StateConfig


@dataclass(frozen=True)
class States:
    """各被试的状态标签与各状态的平均 FC。"""

    labels: np.ndarray        # (被试, 窗口)，0 为整合态
    centroids: np.ndarray     # (被试, 2, 分区, 分区)，Fisher-z
    silhouette: np.ndarray    # (被试,)
    counts: np.ndarray        # (被试,)，实际聚出的状态数


def window_fc(subject_ts):
    """计算每个完整窗口的 Fisher-z 相关矩阵。

    参数:
        subject_ts: 形状为 (时间点, 分区) 的时间序列。

    返回:
        形状为 (窗口数, 分区, 分区) 的数组，对角线置零。

    异常:
        ValueError: 时间序列不足一个完整窗口。
    """
    subject_ts = np.asarray(subject_ts, dtype=float)
    if subject_ts.ndim != 2 or subject_ts.shape[0] < WIN_TR:
        raise ValueError("时间序列必须为帧 × 分区，且至少包含一个完整窗口")
    windows = []
    for start in range(0, subject_ts.shape[0] - WIN_TR + 1, STEP_TR):
        matrix = np.corrcoef(subject_ts[start:start + WIN_TR], rowvar=False)
        matrix = np.arctanh(np.clip(matrix, -1 + 1e-7, 1 - 1e-7))
        np.fill_diagonal(matrix, 0.0)
        windows.append(matrix)
    return np.asarray(windows)


def cartographic(fc, n_bins=N_BINS, seed=SEED):
    """逐窗口统计「参与系数 × 模块内度」的二维直方图，即该窗口的地形图剖面。

    关于 BCT 的随机种子：community_louvain 每次调用都会用 seed 新建一个
    RandomState，所以给所有窗口同一个 seed 会让它们走同一条随机序列。这样能
    复现早期结果，但窗口之间的 Louvain 扰动是相关的，而不是独立的。

    参数:
        fc: 形状为 (窗口数, 分区, 分区) 的 Fisher-z 相关矩阵。
        n_bins: 直方图每个维度的分箱数。
        seed: 逐窗口 Louvain 的随机种子。

    返回:
        形状为 (窗口数, n_bins ** 2) 的剖面，每行和为 1。

    异常:
        ValueError: 某个窗口没有节点落在剖面范围内。
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
        raise ValueError("窗口内有节点落在剖面范围之外")
    return profiles / totals


def _embedding(profiles, metric):
    """把剖面投到单位球面，使欧氏距离的排序等同相关距离。

    sklearn 的 KMeans 不支持相关距离。每行中心化再归一化后，平方欧氏距离等于
    2 - 2r，分配步骤与相关距离同序；质心仍是该空间的线性均值而非球面均值，
    因此是相关 k-means 的近似实现。

    参数:
        profiles: 形状为 (窗口数, 特征数) 的剖面。
        metric: "correlation" 时做嵌入，其余取值原样返回。

    返回:
        用于聚类的特征矩阵。
    """
    if metric != "correlation":
        return profiles
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)


def cluster_states(profiles, config=StateConfig()):
    """用 k-means 把剖面划成两个状态。

    参数:
        profiles: 形状为 (窗口数, 特征数) 的剖面。
        config: 聚类设置。

    返回:
        (窗口标签, 轮廓系数, 实际聚出的状态数)。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(profiles) < 2:
        raise ValueError("状态估计至少需要两个窗口")
    x = _embedding(profiles, config.metric)
    labels = KMeans(n_clusters=2, n_init=config.kmeans_ninit,
                    random_state=config.seed).fit_predict(x)
    count = int(np.unique(labels).size)
    silhouette = (float(silhouette_score(x, labels))
                  if 1 < count < len(labels) else np.nan)
    return labels, silhouette, count


def align_states(labels, profiles, fc, n_bins=N_BINS):
    """重新编号使 0 号为整合态，再按状态平均窗口 FC。

    参数:
        labels: k-means 给出的窗口标签。
        profiles: 形状为 (窗口数, n_bins ** 2) 的剖面。
        fc: 形状为 (窗口数, 分区, 分区) 的窗口 FC。
        n_bins: 剖面的分箱数。

    返回:
        (对齐后的标签, 形状为 (2, 分区, 分区) 的状态质心)。
    """
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
    """估计单个被试的状态。

    参数:
        subject_ts: 形状为 (时间点, 分区) 的时间序列。
        config: 状态估计设置。

    返回:
        (标签, 质心, 轮廓系数, 状态数)。
    """
    fc = window_fc(subject_ts)
    profiles = cartographic(fc, config.n_bins, config.seed)
    labels, silhouette, count = cluster_states(profiles, config)
    labels, centroids = align_states(labels, profiles, fc, config.n_bins)
    return labels, centroids, silhouette, count


def _worker(job):
    index, subject_ts, config = job
    return (index, *one_subject(subject_ts, config))


def estimate(timeseries, config=StateConfig(), workers=1):
    """逐个被试估计状态并堆叠输出。

    只塌缩成一个状态的被试，其质心为 NaN、状态数小于 2。被试名由调用方掌握，
    因此由调用方负责发现并报错。

    参数:
        timeseries: 形状为 (被试, 时间点, 分区) 的时间序列。
        config: 状态估计设置。
        workers: 进程数，1 表示串行。

    返回:
        States 记录，含标签、质心、轮廓系数与各被试的状态数。

    异常:
        ValueError: 没有被试，或 workers 不为正。
    """
    n, frames = timeseries.shape[:2]
    if n == 0:
        raise ValueError("没有被试可供估计")
    if workers < 1:
        raise ValueError("workers 必须为正")
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
    return States(labels=labels, centroids=centroids, silhouette=silhouette, counts=counts)
