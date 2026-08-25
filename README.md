# ATLAS

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Python Version][badge-pyversions]][link-pypi]
[![Codecov][badge-codecov]][link-codecov]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/smilies-polito/atlas-smilies/test.yaml?branch=public
[badge-docs]: https://img.shields.io/readthedocs/atlas-smilies
[badge-pyversions]: https://img.shields.io/pypi/pyversions/atlas-smilies
[badge-codecov]: https://codecov.io/gh/smilies-polito/atlas-smilies/branch/public/graph/badge.svg
[link-codecov]: https://codecov.io/gh/smilies-polito/atlas-smilies
[link-pypi]: https://pypi.org/project/atlas-smilies

ATLAS is a framework for multi-omic trajectory inference from paired single-cell RNA and ATAC sequencing data. By integrating transcriptional and chromatin accessibility information within a unified representation, ATLAS enables pseudotime reconstruction and cell fate prediction that directly incorporate regulatory dynamics.
See our publication to learn more:

```
Leclercq, A., Martini, L., Bardini, R., Savino, A., & Di Carlo, S. (2026). ATLAS: A scverse-compatible package for multi-omic single-cell trajectory inference integration. bioRxiv. https://doi.org/10.64898/2026.05.23.727175

```
![ATLAS workflow](https://raw.githubusercontent.com/smilies-polito/atlas-smilies/public/docs/_static/imgs/workflow.svg)

ATLAS main functionalities include:
- Computation of gene activity from scATAC-seq data
- Support for multi-omics representations using [MuData]
- Creation of a [Weighted Nearest Neighbors] graph
- Trajectory inference based on multiple strategies
- Trajectory evaluation based on unsupervised metrics

## Getting started

Please refer to the [documentation][]
in particular, the [API documentation][].

## Installation

To install ATLAS please use:
```bash
pip install atlas-smilies
```

### Optional components

Three features draw on packages ATLAS does not install. Everything else works without them.

**Fate tree figures** need [scFates][], available as an extra:
```bash
pip install atlas-smilies[trees]
```
Keeping it separate lets ATLAS install without a compiler where
that wheel is unavailable.

**Gene activity from scATAC-seq** (`atlas.pp.compute_gene_activity`) needs `pysam` to read
the fragment file.

**Faster trajectory inference.** `atlas.tl.CellRankExtension.run` defaults to
`method="krylov"`, which needs `petsc4py` and `slepc4py`.

Without them CellRank falls back to `method="brandts"`, which requires a dense
transition matrix. Evaluate conda forge or installation of PETSC and SLEPC.

### Platforms

ATLAS supports Python 3.11 through 3.14 on the platforms below. This table is written
from [`platform-support.toml`](platform-support.toml), which the CI job
`Platforms / Resolution matrix` checks against what actually resolves — in both
directions, so an entry that stops working and one that starts working are equally
reported.

| Platform | Python | Installs from wheels | Verified |
| --- | --- | --- | --- |
| Linux x86_64 | 3.11 – 3.14 | yes | test suite |
| Linux aarch64 | 3.11 – 3.14 | yes, core only — see below | resolution only |
| macOS arm64 | 3.11 – 3.14 | yes | resolution only |
| macOS x86_64 (Intel) | 3.11 – 3.13 | yes, on a pinned older JAX | not yet verified |
| Windows x86_64 | 3.11 – 3.14 | yes | resolution only |

"Resolution only" means a consistent set of wheels exists and has been checked, but no
machine of that platform has run ATLAS. Those platforms are expected to work and are not
yet claimed to.

**Linux aarch64 and the `trees` extra.** `pip install atlas-smilies` needs no compiler
here, but `pip install atlas-smilies[trees]` does: `scikit-misc` publishes no aarch64
Linux wheel, so it is built from source and a Fortran toolchain is required. This is why
scFates is an extra rather than a dependency — the core package stays installable
everywhere.

**macOS x86_64 (Intel).** Python 3.14 does not install: no `jaxlib` new enough for
CPython 3.14 ships a macOS x86_64 wheel, and ATLAS reaches it through
`palantir → mellon → jaxopt`. On 3.11 to 3.13 the resolver finds a complete set of
wheels, but pins `jax` and `jaxlib` to 0.4.38 — considerably older than every other
platform receives. Use Python 3.13 or below on Intel Macs, and consider conda, which
publishes a macOS x86_64 `jaxlib` that PyPI does not.

**Which dependency versions you get depends on your Python.** cellrank 2.1.0 onward
requires Python 3.12, so an installation on 3.11 receives cellrank 2.0.7 and the
contemporaneous scanpy, jax and numpy, while 3.12 and above receive cellrank 2.3.2 and
current versions of the rest. Both are supported; they are not the same environment.

[scFates]: https://scfates.readthedocs.io/
[muon]: https://muon.readthedocs.io/

## Release notes

See the [CHANGELOG][].

## Contact

For questions, bug report and help requests, please use the [issue tracker][].

## Related Works
- Lange, M., Bergen, V., Klein, M. et al. CellRank for directed single-cell fate mapping. Nat Methods 19, 159–170 (2022). https://doi.org/10.1038/s41592-021-01346-6
- Weiler, P., Lange, M., Klein, M. et al. CellRank 2: unified fate mapping in multiview single-cell data. Nat Methods 21, 1196–1205 (2024). https://doi.org/10.1038/s41592-024-02303-9
- Setty, M., Kiseliovas, V., Levine, J. et al. Characterization of cell fate probabilities in single-cell data with Palantir. Nat Biotechnol 37, 451–460 (2019). https://doi.org/10.1038/s41587-019-0068
- Louis Faure, Ruslan Soldatov, Peter V. Kharchenko, Igor Adameyko, scFates: a scalable python package for advanced pseudotime and bifurcation analysis from single cell data, Bioinformatics, btac746; doi: https://doi.org/10.1093/bioinformatics/btac746

## Disclaimer
This project includes third-party code under MIT and BSD-3 licenses, see [THIRD\_PARTY\_NOTICES][] for details.


[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/smilies-polito/atlas-smilies/issues
[tests]: https://github.com/smilies-polito/atlas-smilies/actions/workflows/test.yaml
[documentation]: https://atlas-smilies.readthedocs.io
[CHANGELOG]: https://atlas-smilies.readthedocs.io/en/latest/changelog.html
[api documentation]: https://atlas-smilies.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/atlas-smilies
[codecov]: https://codecov.io/gh/smilies-polito/atlas-smilies
[THIRD\_PARTY\_NOTICES]: https://github.com/smilies-polito/atlas-smilies/blob/public/THIRD_PARTY_NOTICES

[Weighted Nearest Neighbors]: https://www.sciencedirect.com/science/article/pii/S0092867421005833
[MuData]: https://mudata.readthedocs.io/stable/
