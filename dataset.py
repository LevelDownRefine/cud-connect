"""SUDMEX-CONN 可卡因使用障碍研究的样本选择与加载。

本数据相关的一切都在这里：排除名单、协变量名、归档布局与输出命名。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = frozenset({"sub-154", "sub-129", "sub-136"})
COVARIATES = ("age", "sex", "educ_yr", "mean_fd")
DUMMY = 5        # 非稳态卷数，与论文一致
STATES = {"integrated": 0, "segregated": 1}


@dataclass(frozen=True)
class Sample:
    """分析样本的时间序列，已丢弃前几个非稳态卷。"""

    timeseries: np.ndarray    # (被试, 时间点, 分区)
    subjects: list[str]
    parcels: list[str]
    dropped_parcels: int


def read_covariates(root):
    """读取 `<root>/data/covariates.tsv` 中的协变量表。

    参数:
        root: 项目根目录。

    返回:
        协变量 DataFrame。
    """
    return pd.read_csv(Path(root) / "data" / "covariates.tsv", sep="\t", dtype={"sex": str})


def complete_cases(cov, columns=COVARIATES):
    """筛出不在排除名单、且协变量齐全的行。

    参数:
        cov: 协变量 DataFrame。
        columns: 需要检查的协变量列。

    返回:
        (布尔掩码, 各列的数值数组)。
    """
    nums = {c: pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in columns}
    finite = np.isfinite(np.column_stack(list(nums.values()))).all(axis=1)
    return finite & ~cov["subject"].isin(EXCLUDE).to_numpy(), nums


def load_timeseries(root, *, limit=0):
    """按协变量表的顺序加载分析样本。

    参数:
        root: 项目根目录。
        limit: 只取前若干名被试，0 表示不限。

    返回:
        Sample 记录。

    异常:
        ValueError: 选中的被试在归档中缺失。
    """
    cov = read_covariates(root)
    keep, _ = complete_cases(cov)
    cov = cov[keep].reset_index(drop=True)
    archive = np.load(Path(root) / "data" / "timeseries_all.npz")
    values, names = archive["ts"], list(archive["subjects"])
    index = {name: i for i, name in enumerate(names)}
    missing = [s for s in cov["subject"] if s not in index]
    if missing:
        raise ValueError(f"timeseries_all.npz 中缺少这些被试：{missing}")
    if limit:
        cov = cov.iloc[:limit]
    ts = values[[index[s] for s in cov["subject"]]][:, DUMMY:, :].astype(np.float64)
    # 整列全 NaN 的分区会让 Louvain 把每个节点都划成单点，直接剔除。
    parcel_ok = ~np.isnan(ts).any(axis=(0, 1))
    return Sample(
        timeseries=ts[:, :, parcel_ok],
        subjects=list(cov["subject"]),
        parcels=[p for p, ok in zip(archive["parcels"], parcel_ok) if ok],
        dropped_parcels=int((~parcel_ok).sum()),
    )


def sample_rows(cov, avail, columns=COVARIATES):
    """筛出图完整、协变量也完整的行，并取出对应分组。

    参数:
        cov: 协变量 DataFrame。
        avail: 每个被试图是否完整的布尔数组。
        columns: 需要检查的协变量列。

    返回:
        (行索引, 这些行的分组标签, 各列的数值数组)。
    """
    keep, nums = complete_cases(cov, columns)
    rows = np.where(avail.all(axis=1) & keep)[0]
    return rows, cov["group"].to_numpy()[rows], nums


def load_state_fc(root, state):
    """取某个动态状态的平均 FC，并按被试名对齐到协变量表。

    参数:
        root: 项目根目录。
        state: "integrated" 或 "segregated"。

    返回:
        (FC 堆栈, 可用性掩码, 分区名, 协变量表)。
    """
    cov = read_covariates(root)
    archive = np.load(Path(root) / "data" / "dfc_states.npz")
    parcels = list(archive["parcels"])
    cent = archive["centroids"][:, STATES[state]]
    # 归档里只有分析样本，因此按被试名重新对齐。
    position = {s: i for i, s in enumerate(archive["subjects"])}
    fc = np.full((len(cov),) + cent.shape[1:], np.nan)
    hit = cov["subject"].isin(position).to_numpy()
    fc[hit] = cent[[position[s] for s in cov.loc[hit, "subject"]]]
    return fc, np.isfinite(fc[:, 0, 0])[:, None], parcels, cov


def load_static_fc(root):
    """读取窗口平均 FC 矩阵，用于不做状态划分的那条路径。

    参数:
        root: 项目根目录。

    返回:
        (FC 堆栈, 可用性掩码, 分区名, 协变量表)。
    """
    archive = np.load(Path(root) / "data" / "fc_matrices.npz")
    return archive["fc"], archive["available"], list(archive["parcels"]), read_covariates(root)


def output_suffix(state=None, *, alpha_edge=0.05, fisher_z=True):
    """输出文件后缀，避免不同阈值与状态的运行互相覆盖。

    `_noz` 只会出现在不指定状态的分支：状态质心本身已经是 Fisher-z 值。

    参数:
        state: 状态名，None 表示不按状态划分。
        alpha_edge: 边阈值。
        fisher_z: 是否做 Fisher-z 变换。

    返回:
        文件名后缀。
    """
    threshold = "" if alpha_edge == 0.05 else f"_a{alpha_edge:g}".replace(".", "")
    if state:
        return f"_dfc_{state}" + threshold
    return ("" if fisher_z else "_noz") + threshold
