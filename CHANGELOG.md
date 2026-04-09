# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
