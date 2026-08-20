# Changelog

All notable changes to this project will be documented in this file.

## Unreleased
### Added
- `atlas.tl.PalantirExtension.early_cell` selects the initial cell required by `atlas.tl.PalantirExtension.run` from the multiscale diffusion space, so it no longer has to be picked by hand. Selection is delegated to `palantir.utils.early_cell`, which cannot accept a `MuData` directly. The expensive fallback remains opt-in through `fallback_seed` and warns before it runs, since it costs as much as `run` itself.
- `atlas.pp.gene_activity` derives a gene activity modality from scATAC-seq without building any graph, so it can be run, inspected or replaced on its own.
- `atlas.pp.wnn` builds a weighted nearest neighbor graph over **any** two modalities, rather than only gene expression and gene activity. Neighborhood size, number of components and representation accept either a single value for every modality or a per-modality mapping.
- `atlas.pp.knn` builds a nearest neighbor graph over an already integrated representation, such as the output of a joint embedding method, for which no weighting applies.

### Changed
- The temporary `AnnData` objects built to interface with CellRank and Palantir are now constructed in `atlas.tl.utils` rather than inside the extension classes. Internal reorganisation with no change in behaviour.
- `atlas.tl.PalantirExtension` and `atlas.tl.CellRankExtension` now live in `atlas/tl/palantir\_extension.py` and `atlas/tl/cellrank\_extension.py` respectively, replacing `atlas/tl/trajectory\_inference.py`. Both remain importable from `atlas.tl` as before.
- Raised the minimum supported Palantir version from `>1.3` to `>=1.4.5`, to establish a single numerical baseline for seeded results. Results produced against Palantir below `1.4.5` are not reproducible under this floor. Furthermore,  `palantir.utils.early_cell` is guaranteed to forward `eigvec_key` to `palantir.utils.fallback_terminal_cell`.

### Notes
- The new graph entry points annotate the object they are given rather than returning a restricted one, so no modality is removed. Where restricting which modalities take part is required, it happens on a throwaway object.
- The new graph entry points compute **no** embedding, matching the convention that neighbor construction and embedding are separate steps. Trajectory inference reads only the graph and is unaffected; `atlas.pl` defaults to an embedding these routes do not produce.

## Release
### [1.0.0] - 2026-05-25
First public release

## [0.0.5] - 2026-04-01
### Fixed
- Fix aggregation of fate probabilities when `cluster_key` is categorical preventing the inclusion of unused categories
- Fix _safe_mudata MuData construction to make is compatible with mudata > 0.3
- Fix muon.MuData copy mechanism in atlas.pp.preprocessing: if copy = True performs deep copy and preserves all information in the new MuData instance

### Added
- Renamed atlas.tl.PalantirExtension.compute\_diffusion\_map in atlas.tl.PalantirExtension.compute\_diffusion\_maps

### Removed
- Removed atlas.tl.PalantirExtension.compute\_diffusion\_map

## [0.0.6] - 2026-04-09
### Fixed
- Fixed new MuData creation in atlas.pp.preprocessing with safe MuData copy and replacing `del data.mod["atac"]`

### Added
- Added function _safe_mudata

## [0.0.7] - 2026-04-10
### Changed
- Changed KeyError into warning in `atlas.tl.evaluate.terminal_pseudotime_enrichment` when no terminal states are found. In this case np.NaN is returned.


## [0.0.8] - 2026-04-13
### Changed
- Added `count_reads` parameter to atlas.pp.preprocessing. The parameter is used in activity computation using muon.
- Parameter `n_neighbors` in `atlas.pp.preprocessing` updated to include None and adopt standard behavior for `mu.pp.neighbors`.
- Parameter `knn_rna` and `knn_act` in `atlas.pp.neighbors` set to 15 as in `sc.pp.neighbors` default version 1.12.0.
- Parameter `n_pcs_rna` and `n_pcs_act` in `atlas.pp.preprocessing` set to 50 as in `sc.pp.pca` default version 1.12.0.

## [0.0.9] - 2026-04-20
### Changed
- Changed plot size in `atlas.pl.plot_trends` and adjusted for labels and legend.


## [0.0.11] - 2026-05-18
### Added
- Added `backend` parameter to atlas.tl.CellRankExtension.compute\_transition\_matrix. The parameter is used for parallelization.
- Added `backend` parameter to atlas.tl.CellRankExtension.run. The parameter is used for parallelization.

## [0.0.12] - 2026-05-20
### Changed
- Fixed plot fate probabilities when a single fate is found

## [0.1.0] - 2026-05-23
Package release on github
