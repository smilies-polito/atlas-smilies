# ATLAS

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/smilies-polito/ATLAS/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/ATLAS

ATLAS is a multi-omics single-cell trajectory inference framework that integrates gene expression and gene activity to resolve cell fate dynamics.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install ATLAS:

<!--
1) Install the latest release of `ATLAS` from [PyPI][]:

```bash
pip install ATLAS
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/smilies-polito/ATLAS.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

## Disclaimer
This project includes third-party code under MIT and BSD-3 licenses.
See THIRD\_PARTY\_NOTICES for details.

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/smilies-polito/ATLAS/issues
[tests]: https://github.com/smilies-polito/ATLAS/actions/workflows/test.yaml
[documentation]: https://ATLAS.readthedocs.io
[changelog]: https://ATLAS.readthedocs.io/en/latest/changelog.html
[api documentation]: https://ATLAS.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/ATLAS
