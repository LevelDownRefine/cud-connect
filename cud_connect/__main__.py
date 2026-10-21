"""Command line entry point.

    python -m cud_connect states [root] [n_bins] [n_init] [limit] [cpus] [all]
    python -m cud_connect nbs [root] [n_perm] [alpha] [covariates] [z|noz] [state] [all]
    python -m cud_connect all [root] [--cpus N] [--alpha-edge A] [--n-perm N]

`states` and `nbs` keep the positional arguments of the original scripts.
"""

import argparse
import os
from pathlib import Path
import sys

# Cap nested numerical threads before numpy is imported: subjects run in separate processes.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

from . import nbs, states  # noqa: E402
from .config import NbsConfig, StateConfig  # noqa: E402


def _run_all(argv):
    parser = argparse.ArgumentParser(prog="python -m cud_connect all",
                                     description="Run state estimation, then state-specific NBS.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--alpha-edge", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    states.run_states(args.root, StateConfig(cpus=args.cpus))
    for state in ("integrated", "segregated"):
        # State centroids are already Fisher-z values: do not transform again.
        nbs.run_nbs(args.root, NbsConfig(state=state, n_perm=args.n_perm,
                                         alpha_edge=args.alpha_edge,
                                         fisher_z=False, all_parcels=True))


def main(argv=None):
    """Dispatch to a stage; returns an exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args else "--help"
    stages = {"states": states.main, "nbs": nbs.main}
    if stage in ("-h", "--help"):
        print(__doc__)
    elif stage == "all":
        _run_all(args[1:])
    elif stage in stages:
        stages[stage](args[1:])
    else:
        raise SystemExit(f"unknown stage {stage!r}; choose states, nbs or all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
