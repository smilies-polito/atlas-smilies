# ATLAS

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/smilies-polito/atlas-smilies/test.yaml?branch=public
[badge-docs]: https://img.shields.io/readthedocs/atlas-smilies

ATLAS is a framework for multi-omic trajectory inference from paired single-cell RNA and ATAC sequencing data. By integrating transcriptional and chromatin accessibility information within a unified representation, ATLAS enables pseudotime reconstruction and cell fate prediction that directly incorporate regulatory dynamics.
See our publication to learn more:

```
Publication will come soon!
```

![ATLAS WORKFLOW](https://raw.githubusercontent.com/smilies-polito/atlas-smilies/public/imgs/workflow.svg)

## Getting started

Please refer to the [documentation][]
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.

There are several alternative options to install ATLAS:

1) Install the latest release of `ATLAS` from [PyPI][]:

```bash
pip install atlas-smilies
```

1. Install the latest development version:

```bash
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Related Works
- Lange, M., Bergen, V., Klein, M. et al. CellRank for directed single-cell fate mapping. Nat Methods 19, 159–170 (2022). https://doi.org/10.1038/s41592-021-01346-6
- Weiler, P., Lange, M., Klein, M. et al. CellRank 2: unified fate mapping in multiview single-cell data. Nat Methods 21, 1196–1205 (2024). https://doi.org/10.1038/s41592-024-02303-9
- Setty, M., Kiseliovas, V., Levine, J. et al. Characterization of cell fate probabilities in single-cell data with Palantir. Nat Biotechnol 37, 451–460 (2019). https://doi.org/10.1038/s41587-019-0068
- Louis Faure, Ruslan Soldatov, Peter V. Kharchenko, Igor Adameyko, scFates: a scalable python package for advanced pseudotime and bifurcation analysis from single cell data, Bioinformatics, btac746; doi: https://doi.org/10.1093/bioinformatics/btac746

## Disclaimer
This project includes third-party code under MIT and BSD-3 licenses.
See THIRD\_PARTY\_NOTICES for details.


[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/smilies-polito/atlas-smilies/issues
[tests]: https://github.com/smilies-polito/atlas-smilies/actions/workflows/test.yaml
[documentation]: https://atlas-smilies.readthedocs.io
[changelog]: https://atlas-smilies.readthedocs.io/en/latest/changelog.html
[api documentation]: https://atlas-smilies.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/atlas-smilies
