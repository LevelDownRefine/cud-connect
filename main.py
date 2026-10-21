"""SUDMEX-CONN 动态功能连接研究的命令行入口。

    python main.py states [root] [n_bins] [n_init] [limit] [cpus]
    python main.py nbs [root] [n_perm] [alpha] [covariates] [z|noz] [state]
    python main.py all [root] [--cpus N] [--alpha-edge A] [--n-perm N]

算法在 cud_connect 包里；这个文件负责研究侧的事：读哪些文件、校正哪些协变量、
结果写到哪里。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 在导入 numpy 之前压低底层数值线程：每个被试是一个独立进程。
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dataset  # noqa: E402
from cud_connect import nbs as nbs_lib  # noqa: E402
from cud_connect import states as states_lib  # noqa: E402
from cud_connect.config import N_BINS, SEED, STEP_TR, WIN_TR, NbsConfig, StateConfig  # noqa: E402


def run_states(root, config, *, cpus=4, limit=0):
    """估计每个被试的状态，写出归档、质量表与元数据。

    参数:
        root: 项目根目录。
        config: 状态估计设置。
        cpus: 进程数。
        limit: 只取前若干名被试，0 表示不限。

    异常:
        ValueError: 有被试只聚出一个状态。
    """
    start = time.time()
    sample = dataset.load_timeseries(root, limit=limit)
    n, frames, parcels = sample.timeseries.shape
    windows = len(range(0, frames - WIN_TR + 1, STEP_TR))
    print(f"被试 {n}  时间点 {frames}  分区 {parcels}  窗口 {windows}", flush=True)
    result = states_lib.estimate(sample.timeseries, config, workers=cpus)
    single = [s for s, c in zip(sample.subjects, result.counts) if c < 2]
    if single:
        raise ValueError(f"只聚出一个状态的被试：{single}")

    analysis = Path(root) / "data"
    np.savez_compressed(analysis / "dfc_states.npz",
                        subjects=np.array(sample.subjects), labels=result.labels,
                        centroids=result.centroids, parcels=np.array(sample.parcels))
    pd.DataFrame({"subject": sample.subjects, "silhouette": result.silhouette,
                  "n_states": result.counts}).to_csv(
        analysis / "dfc_state_quality.tsv", sep="\t", index=False)
    meta = {
        "n": n, "timepoints": frames, "windows": windows, "parcels": parcels,
        "parcels_dropped_for_nan": sample.dropped_parcels,
        "parcel_source": "all-256",
        "win_tr": WIN_TR, "step_tr": STEP_TR, "dummy_volumes": dataset.DUMMY,
        "louvain_B": "negative_asym", "louvain_repeats": 1, "louvain_seed": "shared",
        "n_bins": config.n_bins, "kmeans_ninit": config.kmeans_ninit,
        "kmeans_metric": config.metric, "seed": config.seed,
        "scope": "state estimation and state-specific connectivity only",
        "elapsed_s": round(time.time() - start, 1),
    }
    (analysis / "dfc_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"已写出 dfc_states.npz、状态质量表与元数据；{time.time()-start:.0f}s")


def run_nbs(root, config, *, state=None, covariates=None, seed=0):
    """在某个状态的平均 FC 上做组间比较，写出汇总、边表与零分布。

    参数:
        root: 项目根目录。
        config: 置换次数与边阈值。
        state: 状态名，None 表示使用窗口平均 FC。
        covariates: 需要校正的协变量列，None 表示默认四列。
        seed: 置换的随机种子。
    """
    covariates = tuple(covariates or dataset.COVARIATES)
    start = time.time()
    if state:
        fc, avail, parcels, cov = dataset.load_state_fc(root, state)
    else:
        fc, avail, parcels, cov = dataset.load_static_fc(root)
    atlas = pd.read_csv(Path(root) / "data" / "parcels_cortex.tsv", sep="\t")
    networks = dict(zip(atlas["name"], atlas["network_label"]))

    rows, groups, nums = dataset.sample_rows(cov, avail, covariates)
    edges = nbs_lib.Edges.complete(len(parcels))
    Y = edges.values(fc, rows)
    print(f"图与协变量均完整：n={len(rows)}"
          f"（HC {int((groups == 'HC').sum())} / CUD {int((groups == 'CUD').sum())}）"
          f"  节点 {edges.n_nodes}  边 {len(edges.index[0])}")
    print(f"边阈值 alpha={config.alpha_edge}   Fisher-z={'on' if config.fisher_z else 'off'}"
          f"   n_perm={config.n_perm}   协变量={covariates}")
    print(f"置换 {config.n_perm} 次...", flush=True)

    result = nbs_lib.compare(Y, (groups == "CUD").astype(float),
                             np.column_stack([nums[c][rows] for c in covariates]),
                             edges, config, seed=seed)
    print(f"t_crit={result.t_crit:.4f}  dof={result.dof}")
    for sign in nbs_lib.SIGNS:
        biggest = max((c.size for c in result.components[sign]), default=0)
        print(f"观测[{'+' if sign > 0 else '-'}]：超阈边 {result.selections[sign].size:4d}"
              f"  分量 {len(result.components[sign]):3d}  最大分量 {biggest:4d} 条边")
    for sign in nbs_lib.SIGNS:
        tag = "+" if sign > 0 else "-"
        print(f"  零分布[{tag}]：中位 {np.median(result.null_max[sign]):.0f}"
              f"  p95 {np.percentile(result.null_max[sign], 95):.0f}"
              f"  max {result.null_max[sign].max():.0f}")

    report = {"n": result.n, "n_hc": result.n_hc, "n_cud": result.n_cud,
              "n_perm": config.n_perm, "alpha_edge": config.alpha_edge,
              "fisher_z": bool(config.fisher_z), "t_crit": result.t_crit,
              "covariates": list(covariates), "nodes": edges.n_nodes,
              "edges": int(len(edges.index[0])), "components": []}

    print(f"\n{'#':>3} {'边数':>6} {'节点':>5} {'sum t':>8} {'p(FWE)':>8}  网络构成")
    edge_rows = []
    for sign in nbs_lib.SIGNS:
        tag = "+" if sign > 0 else "-"
        for k, component in enumerate(sorted(result.components[sign], key=lambda x: -x.size)):
            nodes = edges.nodes_of(component)
            p_fwe = result.p_fwe(sign, component.size)
            counts = pd.Series([networks.get(parcels[i], "subcortical")
                                for i in nodes]).value_counts()
            print(f"{tag}{k + 1:>2} {component.size:>6} {nodes.size:>5} "
                  f"{result.t_values[component].sum():>+8.1f} {p_fwe:>8.4f}  "
                  + " ".join(f"{a}:{v}" for a, v in counts.items()))
            report["components"].append({
                "sign": tag, "n_edges": int(component.size), "n_nodes": int(nodes.size),
                "sum_t": float(result.t_values[component].sum()), "p_fwe": float(p_fwe),
                "networks": {a: int(v) for a, v in counts.items()},
                "nodes": [parcels[i] for i in nodes],
            })
            for e in component:
                i, j = edges.index[0][e], edges.index[1][e]
                edge_rows.append((parcels[i], parcels[j],
                                  networks.get(parcels[i], "subcortical"),
                                  networks.get(parcels[j], "subcortical"),
                                  result.t_values[e], tag))

    suffix = dataset.output_suffix(state, alpha_edge=config.alpha_edge,
                                   fisher_z=config.fisher_z)
    analysis = Path(root) / "data"
    if edge_rows:
        pd.DataFrame(edge_rows, columns=["node_i", "node_j", "network_i", "network_j",
                                         "t", "sign"]) \
            .sort_values("t", key=abs, ascending=False) \
            .to_csv(analysis / f"baseline_nbs_edges{suffix}.tsv", sep="\t", index=False)
    significant = [c for c in report["components"] if c["p_fwe"] < 0.05]
    print(f"\nFWE p<0.05 的分量：{len(significant)}"
          + (f"  最大 {significant[0]['sign']}{significant[0]['n_edges']} 条边"
             f" / {significant[0]['n_nodes']} 节点" if significant else ""))
    (analysis / f"baseline_nbs_summary{suffix}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.save(analysis / f"baseline_nbs_null_maxsize{suffix}.npy",
            np.stack([result.null_max[s] for s in nbs_lib.SIGNS]))
    print(f"已写出 baseline_nbs_summary{suffix}.json / baseline_nbs_edges{suffix}.tsv "
          f"/ baseline_nbs_null_maxsize{suffix}.npy  ({time.time() - start:.0f}s)")


def _stage_states(argv):
    parser = argparse.ArgumentParser(prog="python main.py states",
                                     description="估计整合态与分离态。")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_bins", nargs="?", default=N_BINS, type=int)
    parser.add_argument("n_init", nargs="?", default=300, type=int)
    parser.add_argument("limit", nargs="?", default=0, type=int)
    parser.add_argument("cpus", nargs="?", default=4, type=int)
    parser.add_argument("--metric", choices=["correlation", "euclidean"], default="correlation",
                        help="聚类距离")
    parser.add_argument("--seed", type=int, default=SEED, help="Louvain 与 k-means 的随机种子")
    args = parser.parse_args(argv)
    run_states(args.root,
               StateConfig(n_bins=args.n_bins, kmeans_ninit=args.n_init,
                           metric=args.metric, seed=args.seed),
               cpus=args.cpus, limit=args.limit)


def _stage_nbs(argv):
    parser = argparse.ArgumentParser(prog="python main.py nbs",
                                     description="运行网络基础统计（NBS）。")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_perm", nargs="?", default=10000, type=int)
    parser.add_argument("alpha_edge", nargs="?", default=0.05, type=float)
    parser.add_argument("covariates", nargs="?", default=",".join(dataset.COVARIATES))
    parser.add_argument("z", nargs="?", default="z", help="noz 表示不做 Fisher-z")
    parser.add_argument("state", nargs="?", choices=["integrated", "segregated"])
    args = parser.parse_args(argv)
    run_nbs(args.root, NbsConfig(n_perm=args.n_perm, alpha_edge=args.alpha_edge,
                                 fisher_z=args.z.lower() not in ("0", "no", "noz", "false")),
            state=args.state, covariates=tuple(args.covariates.split(",")))


def _stage_all(argv):
    parser = argparse.ArgumentParser(prog="python main.py all",
                                     description="先估计状态，再跑状态特异的 NBS。")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--alpha-edge", type=float, default=0.01)
    parser.add_argument("--metric", choices=["correlation", "euclidean"], default="correlation",
                        help="聚类距离")
    parser.add_argument("--seed", type=int, default=SEED, help="Louvain 与 k-means 的随机种子")
    args = parser.parse_args(argv)
    if args.cpus < 1:
        parser.error("--cpus 必须为正")
    run_states(args.root, StateConfig(metric=args.metric, seed=args.seed), cpus=args.cpus)
    for state in ("integrated", "segregated"):
        # 状态质心本身已经是 Fisher-z 值，不要再次变换。
        run_nbs(args.root, NbsConfig(n_perm=args.n_perm, alpha_edge=args.alpha_edge,
                                     fisher_z=False),
                state=state)


def main(argv=None):
    """分发到某个阶段，返回退出码。

    参数:
        argv: 参数列表，None 表示取 sys.argv。

    返回:
        退出码。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args else "--help"
    stages = {"states": _stage_states, "nbs": _stage_nbs, "all": _stage_all}
    if stage in ("-h", "--help"):
        print(__doc__)
    elif stage in stages:
        stages[stage](args[1:])
    else:
        raise SystemExit(f"未知阶段 {stage!r}，可选 states、nbs 或 all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
