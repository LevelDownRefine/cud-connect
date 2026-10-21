"""状态估计与 NBS 两个阶段的参数。"""

from dataclasses import dataclass

WIN_TR = 22               # 44 秒窗口
STEP_TR = 1
SEED = 0
PC_RANGE = (0.0, 1.0)     # 参与系数，构造上有界
Z_RANGE = (-3.0, 3.0)     # 模块内度 z 分数，BCT 常用显示范围
N_BINS = 40


@dataclass(frozen=True)
class StateConfig:
    """滑窗状态估计的参数。"""

    n_bins: int = N_BINS
    kmeans_ninit: int = 300
    metric: str = "correlation"    # 论文按相关距离聚类
    seed: int = SEED


@dataclass(frozen=True)
class NbsConfig:
    """网络基础统计的参数。"""

    n_perm: int = 10000
    alpha_edge: float = 0.05
    fisher_z: bool = True
