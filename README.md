# cud-connect

Minimal dynamic functional connectivity analysis in cocaine use disorder.

Parcel time series → sliding-window FC → cartographic profiles → integrated/segregated states → state-specific NBS.

## Run

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run python main.py all
```

Run one stage with `python main.py states` or `python main.py nbs`;
NBS requires the saved state outputs.

Run the small synthetic-data tests with `uv run pytest -q`.

## Layout

`cud_connect/` holds the algorithms and knows nothing about this dataset: `config.py`
the parameters, `states.py` the cartographic profiles and state estimation, `nbs.py`
the network-based statistic. `data.py` is the study side — sample selection, archive
layout and output naming — and `main.py` wires the two together and owns the CLI.

## Data and outputs

`data/` contains XCP-D parcel time series, covariates and network labels derived from [SUDMEX-CONN, ds003346 v1.1.2](https://openneuro.org/datasets/ds003346/versions/1.1.2). Retained sample: 118 participants (54 HC, 64 CUD), 226 parcels, 295 frames.

Outputs are saved in `data/`: state labels and mean FC (`dfc_states_all.npz`), state quality, NBS summaries, edge tables and permutation distributions.

## Method

44-second windows, 2-second steps, 274 windows per participant; 40×40 cartographic profiles; k-means with two clusters, 300 initializations and correlation distance. NBS uses 10,000 permutations, edge alpha 0.01 and age, sex, education and mean-FD covariates. Significance applies to each component as a whole.

Independent implementation based on [Zheng et al. (2025)](https://doi.org/10.3389/fnins.2025.1572463). It runs one seeded Louvain partition per window where the paper runs 100 consensus partitions, with a different sample, atlas and primary threshold. This reproduces the core workflow, not every published setting or result.

## Results

118 participants (54 HC, 64 CUD), 226 parcels, edge alpha 0.01, 10,000 permutations:

| state | component | edges | nodes | p(FWE) |
| --- | --- | --- | --- | --- |
| integrated | positive | 221 | 127 | 0.0229 |
| segregated | positive | 265 | 134 | 0.0209 |

Both components are higher in CUD. No negative component survives (p = 0.28 and 0.24).
