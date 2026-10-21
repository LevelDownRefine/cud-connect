"""Run state estimation and state-specific NBS."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--stage', choices=['all', 'states', 'nbs'], default='all')
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error('--cpus must be positive')
    root = args.root.resolve()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    # Limit nested numerical threads while subjects run in separate processes.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env.setdefault(key, '1')

    def run(name, extra):
        cmd = [sys.executable, str(root / name), *map(str, extra)]
        print('RUN:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, env=env, check=True)

    if args.stage in ('all', 'states'):
        run('states.py', [root, 40, 300, 0, args.cpus, 'all'])
    if args.stage in ('all', 'nbs'):
        for state in ('integrated', 'segregated'):
            # The state centroids are ALREADY Fisher-z values: do not transform again.
            run('nbs.py', [root, 10000, .01, 'age,sex,educ_yr,mean_fd', 'noz', state, 'all'])


if __name__ == '__main__':
    main()
