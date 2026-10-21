"""Command line entry point for the SUDMEX-CONN dynamic connectivity study.

    python main.py states [root] [n_bins] [n_init] [limit] [cpus] [all]
    python main.py nbs [root] [n_perm] [alpha] [covariates] [z|noz] [state] [all]
    python main.py all [root] [--cpus N] [--alpha-edge A] [--n-perm N]

The package holds the algorithms; this file holds the study: which files to read,
which covariables to adjust for, and where the outputs go.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Cap nested numerical threads before numpy is imported: subjects run in separate processes.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as dataset  # noqa: E402
from cud_connect import nbs as nbs_lib  # noqa: E402
from cud_connect import states as states_lib  # noqa: E402
from cud_connect.config import (  # noqa: E402
    N_BINS, SEED, STEP_TR, WIN_TR, NbsConfig, StateConfig)


def run_states(root, config, *, cpus=4, all_parcels=True, limit=0):
    """Estimate each subject's states and write the archive, quality table and metadata."""
    start = time.time()
    sample = dataset.load_timeseries(root, all_parcels=all_parcels, limit=limit)
    n, frames, parcels = sample.timeseries.shape
    windows = len(range(0, frames - WIN_TR + 1, STEP_TR))
    print(f"Subjects {n}; frames {frames}; parcels {parcels}; windows {windows}", flush=True)
    result = states_lib.estimate(sample.timeseries, config, workers=cpus)
    single = [s for s, c in zip(sample.subjects, result.counts) if c < 2]
    if single:
        raise ValueError(f"Single-state subjects: {single}")

    analysis = Path(root) / "data"
    suffix = "_all" if all_parcels else ""
    np.savez_compressed(analysis / f"dfc_states{suffix}.npz",
                        subjects=np.array(sample.subjects), labels=result.labels,
                        centroids=result.centroids, parcels=np.array(sample.parcels))
    pd.DataFrame({"subject": sample.subjects, "silhouette": result.silhouette,
                  "n_states": result.counts}).to_csv(
        analysis / f"dfc_state_quality{suffix}.tsv", sep="\t", index=False)
    meta = {
        "n": n, "timepoints": frames, "windows": windows, "parcels": parcels,
        "parcels_dropped_for_nan": sample.dropped_parcels,
        "parcel_source": "all-256" if all_parcels else "cortical-only",
        "win_tr": WIN_TR, "step_tr": STEP_TR, "dummy_volumes": dataset.DUMMY,
        "louvain_B": "negative_asym", "louvain_repeats": 1, "louvain_seed": "shared",
        "n_bins": config.n_bins, "kmeans_ninit": config.kmeans_ninit,
        "kmeans_metric": config.metric, "seed": config.seed,
        "scope": "state estimation and state-specific connectivity only",
        "elapsed_s": round(time.time() - start, 1),
    }
    (analysis / f"dfc_meta{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved dfc_states{suffix}.npz, state quality and metadata; {time.time()-start:.0f}s")


def run_nbs(root, config, *, state=None, all_parcels=False, covariates=None, seed=0):
    """Compare groups on one state's mean FC and write the summary, edges and null."""
    covariates = tuple(covariates or dataset.COVARIATES)
    start = time.time()
    if state:
        fc, avail, parcels, cov = dataset.load_state_fc(root, state, all_parcels)
    else:
        fc, avail, parcels, cov = dataset.load_static_fc(root)
    labels = pd.read_csv(Path(root) / "data" / "parcels_cortex.tsv", sep="\t")
    networks = dict(zip(labels["name"], labels["network_label"]))

    rows, groups, nums = dataset.sample_rows(cov, avail, covariates)
    edges = nbs_lib.Edges.complete(len(parcels))
    Y = edges.values(fc, rows)
    print(f"完整图 + 协变量完整: n={len(rows)}"
          f"（HC {int((groups == 'HC').sum())} / CUD {int((groups == 'CUD').sum())}）"
          f"  节点 {edges.n_nodes}  边 {len(edges.index[0])}")
    print(f"边阈值 alpha={config.alpha_edge}   Fisher-z={'on' if config.fisher_z else 'off'}"
          f"   n_perm={config.n_perm}   协变量={covariates}")

    result = nbs_lib.compare(Y, (groups == "CUD").astype(float),
                             np.column_stack([nums[c][rows] for c in covariates]),
                             edges, config, seed=seed)
    print(f"t_crit={result.t_crit:.4f}  dof={result.dof}")
    for sign in nbs_lib.SIGNS:
        biggest = max((c.size for c in result.components[sign]), default=0)
        print(f"观测[{'+' if sign > 0 else '-'}]: 超阈边 {result.selections[sign].size:4d}"
              f"  分量 {len(result.components[sign]):3d}  最大分量 {biggest:4d} 条边")
    for sign in nbs_lib.SIGNS:
        tag = "+" if sign > 0 else "-"
        print(f"  零分布[{tag}]: 中位 {np.median(result.null_max[sign]):.0f}"
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
                                   fisher_z=config.fisher_z, all_parcels=all_parcels)
    analysis = Path(root) / "data"
    if edge_rows:
        pd.DataFrame(edge_rows, columns=["node_i", "node_j", "network_i", "network_j",
                                         "t", "sign"]) \
            .sort_values("t", key=abs, ascending=False) \
            .to_csv(analysis / f"baseline_nbs_edges{suffix}.tsv", sep="\t", index=False)
    significant = [c for c in report["components"] if c["p_fwe"] < 0.05]
    print(f"\nFWE p<0.05 的分量: {len(significant)}"
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
                                     description="Estimate integrated/segregated states.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_bins", nargs="?", default=N_BINS, type=int)
    parser.add_argument("n_init", nargs="?", default=300, type=int)
    parser.add_argument("limit", nargs="?", default=0, type=int)
    parser.add_argument("cpus", nargs="?", default=8, type=int)
    parser.add_argument("parcels", nargs="?", choices=["all"])
    parser.add_argument("--metric", choices=["correlation", "euclidean"], default="correlation")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run_states(args.root,
               StateConfig(n_bins=args.n_bins, kmeans_ninit=args.n_init,
                           metric=args.metric, seed=args.seed),
               cpus=args.cpus, all_parcels=args.parcels == "all", limit=args.limit)


def _stage_nbs(argv):
    parser = argparse.ArgumentParser(prog="python main.py nbs",
                                     description="Run the network-based statistic.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("n_perm", nargs="?", default=10000, type=int)
    parser.add_argument("alpha_edge", nargs="?", default=0.05, type=float)
    parser.add_argument("covariates", nargs="?", default=",".join(dataset.COVARIATES))
    parser.add_argument("z", nargs="?", default="z",
                        help='"noz" keeps the values as they are')
    parser.add_argument("state", nargs="?", choices=["integrated", "segregated"])
    parser.add_argument("parcels", nargs="?", choices=["all"])
    args = parser.parse_args(argv)
    run_nbs(args.root, NbsConfig(n_perm=args.n_perm, alpha_edge=args.alpha_edge,
                                 fisher_z=args.z.lower() not in ("0", "no", "noz", "false")),
            state=args.state, all_parcels=args.parcels == "all",
            covariates=tuple(args.covariates.split(",")))


def _stage_all(argv):
    parser = argparse.ArgumentParser(prog="python main.py all",
                                     description="Run state estimation, then state-specific NBS.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--alpha-edge", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    run_states(args.root, StateConfig(), cpus=args.cpus, all_parcels=True)
    for state in ("integrated", "segregated"):
        # State centroids are already Fisher-z values: do not transform again.
        run_nbs(args.root, NbsConfig(n_perm=args.n_perm, alpha_edge=args.alpha_edge,
                                     fisher_z=False),
                state=state, all_parcels=True)


def main(argv=None):
    """Dispatch to a stage; returns an exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args else "--help"
    stages = {"states": _stage_states, "nbs": _stage_nbs, "all": _stage_all}
    if stage in ("-h", "--help"):
        print(__doc__)
    elif stage in stages:
        stages[stage](args[1:])
    else:
        raise SystemExit(f"unknown stage {stage!r}; choose states, nbs or all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
