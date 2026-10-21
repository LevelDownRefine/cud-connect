# cud-connect

Minimal dynamic functional connectivity analysis in cocaine use disorder.

Parcel time series → sliding-window FC → cartographic profiles → integrated/segregated states → state-specific NBS.

## Run

```bash
python run.py --cpus 4
```

Use `--stage states` or `--stage nbs` to run a single stage. NBS requires the saved state outputs.

## Data and outputs

`analysis/` contains XCP-D parcel time series, covariates and network labels derived from [SUDMEX-CONN, ds003346 v1.1.2](https://openneuro.org/datasets/ds003346/versions/1.1.2). Retained sample: 118 participants (54 HC, 64 CUD), 226 parcels, 295 frames.

Outputs are saved in `analysis/`: state labels and mean FC (`dfc_states_all.npz`), state quality, NBS summaries, edge tables and permutation distributions.

## Method

44-second windows, 2-second steps, 274 windows per participant; 40×40 cartographic profiles; k-means with two clusters and 300 initializations. NBS uses 10,000 permutations, edge alpha 0.01 and age, sex, education and mean-FD covariates. Significance applies to each component as a whole.

Independent implementation based on [Zheng et al. (2025)](https://doi.org/10.3389/fnins.2025.1572463). It uses Euclidean k-means and one seeded Louvain partition per window; the paper describes correlation distance and 100 consensus runs, with a different sample, atlas and primary threshold. This reproduces the core workflow, not every published setting or result.
