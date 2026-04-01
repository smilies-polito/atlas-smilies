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
