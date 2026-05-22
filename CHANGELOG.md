# Changelog

All notable changes to this project will be documented in this file.

## Release


## Unreleased

## [0.0.5] - 2026-04-01
### Fixed
- Fix aggregation of fate probabilities when `cluster_key` is categorical preventing the inclusion of unused categories

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
