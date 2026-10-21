"""Small synthetic integration tests; no study data or full analysis runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cud_connect import nbs, states
from cud_connect.config import NbsConfig

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


def test_invalid_inputs_are_rejected():
    """The guards fire with a message that names the problem."""
    edges = nbs.Edges.complete(3)
    values = np.zeros((4, 3))
    group = np.array([0., 0., 1., 1.])
    covariates = np.zeros((4, 1))
    cases = {
        "non-finite": (np.full((4, 3), np.nan), group, covariates),
        "coded": (values, np.array([0., 1., 2., 1.]), covariates),
        "terms": (values, group, np.zeros((4, 3))),
    }
    for pattern, (Y, g, Z) in cases.items():
        with pytest.raises(ValueError, match=pattern):
            nbs.compare(Y, g, Z, edges, NbsConfig(n_perm=5))


def test_window_fc_returns_fisher_z_windows():
    """Every window is symmetric with a zero diagonal, and one window per step."""
    rng = np.random.default_rng(3)
    fc = states.window_fc(rng.normal(size=(40, 5)))
    assert fc.shape == (40 - states.WIN_TR + 1, 5, 5)
    np.testing.assert_allclose(fc, np.transpose(fc, (0, 2, 1)), atol=1e-12)
    np.testing.assert_array_equal(np.diagonal(fc, axis1=1, axis2=2), 0.0)


def test_align_states_labels_the_integrated_state_zero():
    """`integrated` is decided by the mean participation coefficient, not by the
    cluster number KMeans happened to hand out."""
    n_bins = 4
    profiles = np.zeros((4, n_bins, n_bins))
    profiles[0, -1, 0] = profiles[1, -1, 0] = 1.0    # windows labelled 1 sit at high PC
    profiles[2, 0, 0] = profiles[3, 0, 0] = 1.0      # windows labelled 0 sit at low PC
    labels, _ = states.align_states(np.array([1, 1, 0, 0]), profiles.reshape(4, -1),
                                    np.zeros((4, 2, 2)), n_bins)
    assert labels.tolist() == [0, 0, 1, 1]


def test_components_splits_disconnected_edges():
    """Two separate triangles give two components; no edges give none."""
    edges = nbs.Edges.complete(6)
    lookup = {pair: i for i, pair in enumerate(zip(*edges.index))}
    triangles = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    sel = np.array([lookup[pair] for pair in triangles])
    assert sorted(c.size for c in nbs.components(sel, edges)) == [3, 3]
    assert nbs.components(np.array([], dtype=int), edges) == []
