"""Small synthetic integration tests; no study data or full analysis runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]


def run_stage(stage, root, *args):
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(REPO / "main.py"), stage, str(root), *map(str, args)],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_state_windows_and_centroids_follow_subject_ids(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(7)
    ts = rng.normal(size=(2, 80, 8))
    for subject in ts:
        latent = rng.normal(size=(80, 3))
        subject[:40, :4] += 2 * latent[:40, :1]
        subject[:40, 4:] += 2 * latent[:40, 1:2]
        subject[40:] += 2 * latent[40:, 2:3]
    subjects = np.array(["sample-a", "sample-b"])
    np.savez(data_dir / "timeseries_all.npz", ts=ts, subjects=subjects,
             parcels=np.array([f"roi-{i}" for i in range(8)]))
    pd.DataFrame({"subject": subjects[::-1], "group": ["CUD", "HC"],
                  "age": [30, 35], "sex": [1, 2], "educ_yr": [12, 14],
                  "mean_fd": [.1, .2]}).to_csv(
        data_dir / "covariates.tsv", sep="\t", index=False)

    run_stage("states", tmp_path, 8, 5, 0, 1)

    with np.load(data_dir / "dfc_states.npz") as output:
        assert output["subjects"].tolist() == subjects[::-1].tolist()
        assert output["labels"].shape == (2, 54)  # 80 - 5 discarded - 22 + 1
        for row, subject in enumerate(output["subjects"]):
            labels = output["labels"][row]
            assert set(labels) == {0, 1}
            series = ts[np.flatnonzero(subjects == subject)[0], 5:]
            windows = []
            for start in range(54):
                correlation = np.corrcoef(series[start:start + 22], rowvar=False)
                np.fill_diagonal(correlation, 0)
                windows.append(np.arctanh(correlation))
            for state in (0, 1):
                expected = np.mean(np.array(windows)[labels == state], axis=0)
                np.testing.assert_allclose(output["centroids"][row, state], expected,
                                           atol=5e-7, rtol=5e-6)
    meta = json.loads((data_dir / "dfc_meta.json").read_text())
    assert (meta["n"], meta["timepoints"], meta["windows"], meta["parcels"]) == (2, 75, 54, 8)


@pytest.mark.parametrize("signal", [0, 1], ids=["null", "signed-components"])
def test_nbs_signed_components_and_adjusted_statistics(tmp_path, signal):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(21)
    n = 30
    subjects = np.array([f"sample-{i:02d}" for i in range(n)])
    parcels = np.array([f"roi-{i}" for i in range(5)])
    group = np.repeat([0, 1], n // 2)
    cov = pd.DataFrame({"subject": subjects, "group": np.where(group, "CUD", "HC"),
                        "age": rng.uniform(20, 45, n), "sex": rng.integers(1, 3, n),
                        "educ_yr": rng.uniform(8, 18, n), "mean_fd": rng.uniform(.05, .3, n)})
    design = np.column_stack([np.ones(n), group, cov[["age", "sex", "educ_yr", "mean_fd"]]])
    upper = np.triu_indices(5, 1)
    noise = rng.normal(0, .2, size=(n, len(upper[0])))
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    known = {(0, 1): 1, (1, 2): 1, (2, 3): -1, (3, 4): -1}
    effects = np.array([known.get(pair, 0) for pair in zip(*upper)])
    # Values above 1 are valid Fisher-z FC, and must not be transformed again.
    values = 1.2 + .02 * cov.age.to_numpy()[:, None] + noise + signal * 1.7 * group[:, None] * effects
    matrices = np.zeros((n, 2, 5, 5))
    matrices[:, 0, upper[0], upper[1]] = values
    matrices[:, 0, upper[1], upper[0]] = values
    order = rng.permutation(n)
    np.savez(data_dir / "dfc_states.npz", subjects=subjects[order],
             parcels=parcels, centroids=matrices[order])
    cov.iloc[rng.permutation(n)].to_csv(data_dir / "covariates.tsv", sep="\t", index=False)
    pd.DataFrame({"name": parcels, "network_label": ["Default"] * 5}).to_csv(
        data_dir / "parcels_cortex.tsv", sep="\t", index=False)

    run_stage("nbs", tmp_path, 19, .01, "age,sex,educ_yr,mean_fd", "noz", "integrated")

    suffix = "_dfc_integrated_a001"
    report = json.loads((data_dir / f"baseline_nbs_summary{suffix}.json").read_text())
    null = np.load(data_dir / f"baseline_nbs_null_maxsize{suffix}.npy")
    assert (report["n"], report["n_hc"], report["n_cud"]) == (30, 15, 15)
    assert report["fisher_z"] is False
    assert null.shape == (2, 19)
    if not signal:
        assert report["components"] == []
        assert not (data_dir / f"baseline_nbs_edges{suffix}.tsv").exists()
        return

    components = {component["sign"]: component for component in report["components"]}
    assert len(report["components"]) == 2
    assert set(components["+"]["nodes"]) == {"roi-0", "roi-1", "roi-2"}
    assert set(components["-"]["nodes"]) == {"roi-2", "roi-3", "roi-4"}
    for row, sign in enumerate(("+", "-")):
        assert components[sign]["n_edges"] == 2
        assert components[sign]["p_fwe"] == (1 + np.count_nonzero(null[row] >= 2)) / 20

    # Independent full-design OLS reference (production uses partial regression).
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    variance = np.sum((values - design @ beta) ** 2, axis=0) / (n - design.shape[1])
    expected_t = beta[1] / np.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1])
    expected_t = dict(zip(zip(upper[0], upper[1]), expected_t))
    edges = pd.read_csv(data_dir / f"baseline_nbs_edges{suffix}.tsv", sep="\t")
    assert len(edges) == 4
    for edge in edges.itertuples():
        pair = (int(edge.node_i[4:]), int(edge.node_j[4:]))
        np.testing.assert_allclose(edge.t, expected_t[pair], rtol=1e-8, atol=1e-8)
