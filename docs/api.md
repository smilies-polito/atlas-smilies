# API

## Preprocessing

Preprocessing utilities for multimodal single-cell data based on
[scanpy](https://scanpy.readthedocs.io) and
[muon](https://muon.readthedocs.io)

```{eval-rst}
.. module:: atlas.pp
.. currentmodule:: atlas

.. autosummary::
    :toctree: generated

    pp.preprocessing
    pp.compute_gene_activity
    pp.wnn
    pp.knn
```

## Tools

```{eval-rst}
.. module:: atlas.tl
.. currentmodule:: atlas

.. autosummary::
    :toctree: generated

    tl.PalantirExtension
    tl.CellRankExtension
    tl.pearson_correlation
    tl.spearman_correlation
    tl.fate_concentration_index
    tl.terminal_pseudotime_enrichment
    tl.terminal_state_silhouette
```

## Plotting

```{eval-rst}
.. module:: atlas.pl
.. currentmodule:: atlas

.. autosummary::
    :toctree: generated

    pl.plot_tree
    pl.plot_trends
    pl.plot_fate_probabilities
    pl.plot_embedding



```
