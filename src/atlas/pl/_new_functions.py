import os
import warnings
from collections.abc import Sequence
from math import ceil
from typing import Literal

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from muon import MuData

from atlas.tl import MultiLineageGAM

from .utils import _fate_frame, _require_scfates, _resolve_basis, _resolve_color, _state_colors

_DEFAULT_LINEAGE_COLOR = "grey"
_UNDECIDED_COLOR = "lightgrey"
_TIME_PRODUCER = "`atlas.tl.PalantirExtension.run` or `atlas.tl.CellRankExtension.compute_kernel`"
_LINEAGE_KEY, _STATES_KEY = "lineages_fwd", "term_states_fwd"
_TREE_COLUMNS = frozenset({"t", "t_sd", "seg", "edge", "milestones"})


def embedding(
    mudata: MuData,
    basis: str = "umap",
    color: str | Sequence[str] | None = None,
    *,
    use_raw: bool | None = None,
    layer: str | None = None,
    **kwargs,
) -> Axes | list[Axes] | None:
    """Plot cells in a stored embedding, coloured by observations or by measured features.

    The multi-modal object is drawn directly, by :func:`muon.pl.embedding`. A colour key is
    resolved against ``mudata.obs`` and against the features of every modality,

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying an embedding in ``.obsm``.
    basis
        Name of the embedding in ``mudata.obsm``, with or without the ``X_`` prefix under
        which embeddings are stored. An embedding inside a modality is named
        ``"<modality>:<basis>"``. Defaults to the embedding :func:`atlas.tl.umap` produces.
    color
        Key, or keys, to colour by. Each names a column of ``mudata.obs`` or a feature of
        one of the modalities, optionally qualified as ``"<modality>:<feature>"``. When
        ``None`` the cells are drawn uncoloured.
    use_raw
        Whether a feature is read from the ``.raw`` attribute of the modality it belongs to,
        as in :func:`muon.pl.embedding`. When ``None`` it is used if present and no valid
        ``layer`` is given.
    layer
        Layer of the modality a feature is read from, as in :func:`muon.pl.embedding`.
    **kwargs
        Additional keyword arguments passed to :func:`scanpy.pl.embedding`.

    Returns
    -------
    The axes drawn on, a list of them when several keys are given, or ``None`` when the
    figure is shown rather than returned.

    Raises
    ------
    KeyError
        If ``basis`` names no embedding on the object, or if a colour key names neither an
        observation nor a feature of any modality.

    Notes
    -----
    A colour key is resolved against ``mudata.obs`` before the modalities, so a name
    occurring in both is taken from ``.obs``. Qualifying it as ``"<modality>:<feature>"``
    selects the feature instead.


    Examples
    --------
    >>> atlas.pl.embedding(mudata, color="pseudotime")
    >>> atlas.pl.embedding(mudata, color=["pseudotime", "shannon_entropy"], ncols=2)
    >>> atlas.pl.embedding(mudata, color=["GATA1", "activity:GATA1"])
    """
    _resolve_basis(mudata, basis)

    if color is not None:
        _resolve_color(mudata, color, use_raw)

    return mu.pl.embedding(mudata, basis=basis, color=color, use_raw=use_raw, layer=layer, **kwargs)


def trends(
    mudata: MuData,
    ptf: str,
    genes: str | Sequence[str],
    *,
    time_key: str = "pseudotime",
    fate_probability_key: str = "fate_probabilities",
    lineages: str | Sequence[str] | None = None,
    n_splines: int = 8,
    n_points: int = 200,
    ncols: int = 2,
    sharex: bool = False,
    figsize: tuple[float, float] | None = None,
    return_models: bool = False,
    show: bool | None = None,
    save: str | None = None,
) -> list[Axes] | dict:
    """Plot lineage-specific dynamics of transcription factor expression and gene activity.

    Trends are fit using :class:`~atlas.tl.MultiLineageGAM`. For each plot, the function displays the fitted dynamics together with its confidence.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying pseudotime, fate probabilities, and the
        ``"rna"`` and ``"activity"`` modalities.
    ptf
        Name of the transcription factor in the ``"rna"`` modality.
    genes
        Name, or names, of the genes in the ``"activity"`` modality.
    time_key
        Key in ``mudata.obs`` containing pseudotime values.
    fate_probability_key
        Key in ``mudata.obsm`` containing the fate probabilities. Its columns name the
        lineages.
    lineages
        Subset of lineages to draw. When ``None`` every lineage that could be fitted is drawn.
    n_splines
        Number of splines used in each GAM.
    n_points
        Number of points at which each curve is evaluated.
    ncols
        Number of panels placed side by side. ``ncols=1`` stacks them in a single column,
        which for one gene is the arrangement :func:`~atlas.pl.plot_trends` produces.
    sharex
        Whether the panels share their pseudotime axis. They are independent by default.
    figsize
        Size of the figure. When ``None`` it is derived from the number of panels.
    return_models
        Whether to return the fitted models rather than the axes.
    show
        Whether to show the figure. When ``None`` the figure is shown unless something is
        being returned.
    save
        Filename to save the figure under, in ``figures/``.

    Returns
    -------
    The axes drawn on, or the fitted models when ``return_models`` is set — keyed by lineage,
    the arrangement :class:`~atlas.tl.MultiLineageGAM` exposes.

    Raises
    ------
    KeyError
        If the factor, a gene, ``time_key`` or ``fate_probability_key`` is absent.
    ValueError
        If the object records no lineage, if a named lineage is not among those recorded, or
        if every lineage was skipped for having negligible weight. The three are reported
        apart, an object that records nothing being a different matter from a mistyped name.

    Examples
    --------
    >>> atlas.pl.trends(mudata, ptf="GATA1", genes="KLF1")
    >>> atlas.pl.trends(mudata, ptf="GATA1", genes=["KLF1", "HBB"], ncols=1)
    >>> models = atlas.pl.trends(mudata, ptf="GATA1", genes="KLF1", return_models=True)
    """
    modalities = ["rna", "activity"]
    missing = [mod for mod in modalities if mod not in mudata.mod]
    if missing:
        raise KeyError(f"missing required modalities in MuData: {', '.join(missing)}")

    genes = [genes] if isinstance(genes, str) else list(genes)

    if fate_probability_key not in mudata.obsm:
        raise KeyError(f"{fate_probability_key} not in mudata.obsm")
    recorded = list(mudata.obsm[fate_probability_key].columns)
    if not recorded:
        raise ValueError(
            f"`mudata.obsm['{fate_probability_key}']` records no lineage, so there is nothing "
            f"to draw; run a trajectory inference first."
        )

    model = MultiLineageGAM(
        mudata=mudata,
        ptf=ptf,
        genes=genes,
        time_key=time_key,
        fate_prob_key=fate_probability_key,
        n_splines=n_splines,
    )
    model.fit()
    model.predict(n_points=n_points)
    predictions = model.predictions

    if not predictions:
        raise ValueError(
            f"every lineage ({', '.join(map(str, recorded))}) was skipped for having negligible "
            f"weight, so there is nothing to draw. The object does record lineages; no cell "
            f"transitions towards any of them."
        )

    if lineages is None:
        drawn = list(predictions)
    else:
        drawn = [lineages] if isinstance(lineages, str) else list(lineages)
        unknown = [name for name in drawn if name not in predictions]
        if unknown:
            raise ValueError(
                f"no lineage called {', '.join(map(repr, unknown))}; those available are "
                f"{', '.join(map(repr, predictions))}."
            )

    colors = _state_colors(mudata, "terminal_states")

    n_panels = 1 + len(genes)
    ncols = max(1, min(ncols, n_panels))
    nrows = ceil(n_panels / ncols)
    if figsize is None:
        figsize = (5.0 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, squeeze=False)
    axes = list(axes.ravel())

    for lineage in drawn:
        pred = predictions[lineage]
        t = pred["t_grid"]
        color = colors.get(lineage, _DEFAULT_LINEAGE_COLOR)

        axes[0].plot(t, pred["gex"], color=color, lw=2, label=lineage)
        axes[0].fill_between(t, pred["gex_lower"], pred["gex_upper"], color=color, alpha=0.2)

        for position, gene in enumerate(genes, start=1):
            curve = pred["genes"][gene]
            axes[position].plot(t, curve["act"], color=color, lw=2, label=lineage)
            axes[position].fill_between(t, curve["act_lower"], curve["act_upper"], color=color, alpha=0.2)

    axes[0].set_ylabel(f"{ptf} expression (z-score)")
    axes[0].set_title(ptf)
    for position, gene in enumerate(genes, start=1):
        axes[position].set_ylabel(f"{gene} activity (z-score)")
        axes[position].set_title(gene)
    for ax in axes[:n_panels]:
        ax.set_xlabel(time_key)

    for ax in axes[n_panels:]:
        ax.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    if save is not None:
        figure_path = os.path.join(os.getcwd(), "figures")
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(os.path.join(figure_path, f"trends_{save}.png"), bbox_inches="tight", dpi=300)

    if show is None:
        show = not return_models
    if show:
        plt.show()

    return model.models if return_models else axes[:n_panels]


def fate_probabilities(
    mudata: MuData,
    *,
    basis: str = "umap",
    fate_probability_key: str = "fate_probabilities",
    lineages: str | Sequence[str] | None = None,
    title: str | None = None,
    legend_loc: str | None = "right margin",
    show: bool | None = None,
    save: str | None = None,
    **kwargs,
) -> Axes | None:
    # This function is adapted from CellRank (BSD 3-Clause License).
    # Original source: https://github.com/scverse/cellrank
    # Copyright (c) 2019, Theis Lab
    # Modifications:
    # - Does not use the Lineage class
    # - Only focuses on plotting fate probabilities and not time.
    # - Uses original scvelo plot function with custom cmap to plot fate probabilities
    # - Controls for singleton and nans
    """Plot fate probabilities on an embedding, blending each cell's two most likely fates.

    This function exploits :func:`scvelo.pl.scatter`.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying fate probabilities and an embedding.
    basis
        Name of the embedding in ``mudata.obsm``.
    fate_probability_key
        Key in ``mudata.obsm`` holding the probabilities. Its columns name the fates.
    lineages
        Subset of fates to draw. When ``None`` every recorded fate is drawn.
    title
        Title of the figure.
    legend_loc
        Where the fates are named, as in :func:`scvelo.pl.scatter`.
    show
        Whether to show the figure, as in :func:`scvelo.pl.scatter`.
    save
        Filename to save the figure under, in ``figures/``, following as in :func:`scvelo.pl.scatter`.
    **kwargs
        Additional keyword arguments passed to :func:`scvelo.pl.scatter`.

    Returns
    -------
    The axes drawn on, or ``None`` when the figure is shown or nothing could be drawn.

    Raises
    ------
    KeyError
        If ``basis`` names no embedding, if ``fate_probability_key`` names no probabilities, or
        if ``lineages`` names a fate the object does not record.

    Warns
    -----
    UserWarning
        If any probability is not a number, in which case no figure is produced.

    Examples
    --------
    >>> atlas.pl.fate_probabilities(mudata)
    >>> atlas.pl.fate_probabilities(mudata, lineages=["Ery", "Mk"])
    """
    import scvelo as scv

    _resolve_basis(mudata, basis)
    probabilities = _fate_frame(mudata, fate_probability_key)
    recorded = [str(name) for name in probabilities.columns]

    if lineages is None:
        selected = recorded
    else:
        selected = [lineages] if isinstance(lineages, str) else [str(name) for name in lineages]
        missing = [name for name in selected if name not in recorded]
        if missing:
            available = ", ".join(recorded) or "none"
            raise KeyError(
                f"{missing} not among the recorded fates ({available}); "
                f"`lineages` names fates to draw, not fates to create"
            )

    if probabilities.isna().to_numpy().any():
        affected = int(probabilities.isna().any(axis=1).sum())
        warnings.warn(
            f"WARNING: {affected} cell(s) carry a probability that is not a number in "
            f"`mudata.obsm['{fate_probability_key}']`; no figure is drawn. Such a cell would be "
            f"assigned to the fate whose probability is missing, `numpy.argmax` treating NaN as "
            f"the largest value, and drawn indistinguishably from one whose fate was determined.",
            stacklevel=2,
        )
        return None

    colors = _state_colors(mudata, "terminal_states")
    palette = [colors.get(name, _DEFAULT_LINEAGE_COLOR) for name in selected]

    embedding_key = basis if basis in mudata.obsm else f"X_{basis}"
    adata = AnnData(X=np.zeros((mudata.n_obs, 1)), obs=mudata.obs.copy())
    adata.obsm[embedding_key] = np.asarray(mudata.obsm[embedding_key])
    scv_basis = embedding_key[2:] if embedding_key.startswith("X_") else embedding_key

    kwargs.setdefault("show", show)
    kwargs.setdefault("save", save)
    kwargs.setdefault("legend_loc", legend_loc)

    if not selected:
        return scv.pl.scatter(adata, basis=scv_basis, color=_UNDECIDED_COLOR, title=title or "", **kwargs)

    if len(selected) == 1:
        name = selected[0]
        column = f"_atlas_p_{name}"
        adata.obs[column] = probabilities[name].to_numpy(dtype=float)
        kwargs.setdefault(
            "color_map", LinearSegmentedColormap.from_list(f"_atlas_{name}", [_UNDECIDED_COLOR, palette[0]])
        )
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 1.0)
        return scv.pl.scatter(adata, basis=scv_basis, color=column, title=title or name, **kwargs)

    return scv.pl.scatter(
        adata,
        basis=scv_basis,
        color_gradients=probabilities[selected],
        palette=palette,
        title=title or "",
        **kwargs,
    )


def _tree_root(mudata: MuData, tree: AnnData, root: str | None) -> int:
    """The principal node the tree runs away from, from the state the object records as initial.

    The choice decides which fate reads as the origin and which read as outcomes
    It is therefore never guessed.
    """
    from atlas.tl.utils import _states_mapping

    states = _states_mapping(mudata, "initial_states")

    if not states:
        raise KeyError(f"mudata records no initial state. Run {_TIME_PRODUCER} to compute one")

    if root is None:
        if len(states) > 1:
            available = ", ".join(sorted(states))
            raise ValueError(
                f"mudata records more than one initial state ({available}); pass `root` naming "
                f"the one to anchor the tree. The choice decides which fate reads as the origin "
                f"and which read as outcomes"
            )
        root = next(iter(states))
    elif root not in states:
        available = ", ".join(sorted(states)) or "none"
        raise KeyError(
            f"'{root}' is not among the recorded initial states ({available}); "
            f"`root` names the state that anchors the tree, not one to create"
        )

    cells = list(states[root])
    positions = tree.obs_names.get_indexer(cells)
    if (positions < 0).any():
        # `get_indexer` reports -1 for a label it cannot find, and -1 indexes the last row.
        missing = [cell for cell, position in zip(cells, positions, strict=True) if position < 0]
        raise KeyError(
            f"initial state '{root}' names {len(missing)} cell(s) mudata no longer holds "
            f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}); an object subset "
            f"since its states were recorded is a different object"
        )

    return int(tree.obsm["X_R"][positions, :].mean(axis=0).argmax())


def _reject_unfittable(probabilities: pd.DataFrame, key: str) -> None:
    values = probabilities.to_numpy(dtype=float)
    names = [str(name) for name in probabilities.columns]

    def _named(mask_by_fate) -> str:
        return ", ".join(name for name, hit in zip(names, mask_by_fate, strict=True) if hit)

    unfinite = ~np.isfinite(values)
    if unfinite.any():
        raise ValueError(
            f"mudata.obsm['{key}'] holds values that are not finite for {_named(unfinite.any(axis=0))} "
            f"({int(unfinite.any(axis=1).sum())} cell(s)); the structure is fitted on the fates "
            f"normalised by their averages, which carries one such value into every cell, so what "
            f"would be fitted is not the recorded probabilities"
        )

    negative = values < 0
    if negative.any():
        raise ValueError(
            f"mudata.obsm['{key}'] holds negative probabilities for {_named(negative.any(axis=0))} "
            f"({int(negative.any(axis=1).sum())} cell(s)); a fate probability is the share of a "
            f"cell heading towards that fate and has no reading below zero"
        )

    # Checked after the sign, a column being able to sum to zero by cancellation otherwise.
    dead = values.sum(axis=0) <= 0
    if dead.any():
        raise ValueError(
            f"mudata.obsm['{key}'] records {_named(dead)} with zero probability for every cell; "
            f"the fates are placed around a circle and each is divided by its average, so a fate "
            f"no cell heads towards has no position to hold. Recompute the probabilities, or "
            f"record only the fates cells head towards"
        )


def _fit_tree(
    mudata: MuData,
    probabilities: pd.DataFrame,
    basis: str,
    time_key: str,
    root: str | None,
    nodes: int,
    method: str,
    ppt_lambda: int,
    random_state: int,
    tree_kwargs: dict,
) -> AnnData:
    """Fit the principal tree over the fate probabilities, on a throwaway object.

    `scFates` cannot consume a ``MuData``, and everything it computes lands on the object it is
    handed; that object is what `return_tree` gives back rather than something discarded.
    """
    scf = _require_scfates("`atlas.pl.fate_tree`")

    names = [str(name) for name in probabilities.columns]
    colors = _state_colors(mudata, "terminal_states")
    palette = [colors.get(name, _DEFAULT_LINEAGE_COLOR) for name in names]

    embedding_key = basis if basis in mudata.obsm else f"X_{basis}"
    tree = AnnData(X=np.zeros((mudata.n_obs, 1)), obs=mudata.obs.copy())
    tree.obsm[f"X_{basis}"] = np.asarray(mudata.obsm[embedding_key])

    tree.obsm[_LINEAGE_KEY] = probabilities.to_numpy(dtype=np.float64)
    tree.obs[_STATES_KEY] = pd.Categorical(np.asarray(names)[probabilities.to_numpy().argmax(axis=1)], categories=names)
    tree.uns[f"{_STATES_KEY}_colors"] = palette

    scf.tl.cellrank_to_tree(
        adata=tree,
        time=time_key,
        Nodes=nodes,
        method=method,
        ppt_lambda=ppt_lambda,
        auto_root=False,
        reassign_pseudotime=False,
        key_cellrank=_LINEAGE_KEY,
        copy=False,
        seed=random_state,
        **tree_kwargs,
    )

    scf.tl.root(tree, _tree_root(mudata, tree, root))
    scf.tl.pseudotime(tree, seed=random_state, copy=False)
    scf.tl.dendrogram(tree)
    # `scFates.tl.dendrogram` opens a figure when none is current. Closing it here
    plt.close()

    return tree


def fate_tree(
    mudata: MuData,
    *,
    basis: str = "umap",
    fate_probability_key: str = "fate_probabilities",
    time_key: str = "pseudotime",
    root: str | None = None,
    color: str | None = None,
    nodes: int = 300,
    method: Literal["ppt", "epg"] = "ppt",
    ppt_lambda: int = 100,
    tree: AnnData | None = None,
    return_tree: bool = False,
    ax: Sequence[Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool | None = None,
    save: str | None = None,
    random_state: int = 42,
    tree_kwargs: dict | None = None,
    graph_kwargs: dict | None = None,
    dendrogram_kwargs: dict | None = None,
) -> list[Axes] | AnnData:
    # This function adapts a tutorial from scFates (BSD 3-Clause).
    # Original source: https://scfates.readthedocs.io/en/latest/Conversion_from_CellRank_pipeline.html
    # Copyright (c) 2020, LouisFaure
    # Modifications:
    # - Adapted the code to MuData object resulting from ATLAS
    # - Constructs instance of the Lineage class from CellRank to run tree inference
    """Plot the branching structure implied by fate probabilities, using :cite:`scfates`.

    A principal tree is fitted over the fate probabilities and return on a single visualization
    the projected tree onto an embedding, and as a dendrogram.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying fate probabilities, a pseudotime and an
        embedding.
    basis
        Name of the embedding in ``mudata.obsm``, with or without the ``X_`` prefix under which
        embeddings are stored.
    fate_probability_key
        Key in ``mudata.obsm`` holding the probabilities. Its columns name the fates.
    time_key
        Column of ``mudata.obs`` holding the pseudotime the tree is fitted against.
    root
        Which recorded initial state anchors the tree. When the object records one it is used
        and this may be left unset; when it records several, one must be named.
    color
        Column of ``mudata.obs`` the cells are coloured by, in both views.
    nodes
        Number of nodes composing the principal tree.
    method
        Tree inference method, one of ``{'ppt', 'epg'}``.
    ppt_lambda
        Penalty on tree length, for ``method='ppt'``.
    tree
        A tree returned earlier by ``return_tree``. When given, it is drawn as it is and nothing
        is fitted. It must have been fitted on the cells ``mudata`` currently holds.
    return_tree
        Whether to return the fitted tree instead of the axes.
    ax
        The two axes to draw into, in order: the embedding, then the dendrogram. When given, no
        figure is created.
    figsize
        Size of the figure, when one is created.
    show
        Whether to show the figure. Defaults to showing it unless ``return_tree`` is set.
    save
        Filename to save the figure under, in ``figures/``.
    random_state
        Seed for the fit.
    tree_kwargs
        Additional keyword arguments for :func:`scFates.tl.tree`, including the ``epg_*``
        parameters when ``method='epg'``.
    graph_kwargs
        Additional keyword arguments for :func:`scFates.pl.graph`.
    dendrogram_kwargs
        Additional keyword arguments for :func:`scFates.pl.dendrogram`.

    Returns
    -------
    The two axes drawn on, or the fitted tree when ``return_tree`` is set. What that tree
    carries is described under Notes.

    Raises
    ------
    KeyError
        If ``basis``, ``fate_probability_key``, ``time_key`` or ``color`` names nothing the
        object records, if ``root`` names no recorded initial state, or if an initial state
        names cells the object no longer holds.
    ValueError
        If fewer than two fates are recorded, if several initial states are recorded and none
        is named, if ``ax`` is not exactly two axes, or if ``tree`` was fitted on other cells.
        If any recorded probability is not finite or is negative, or if a recorded fate carries
        zero probability for every cell — see Notes.

    Notes
    -----
    Below three fates the representation is ``[P(fate₀), pseudotime]`` and is exact,
    one probability determining the other. At three or more it is the circular
    projection of :func:`cellrank.pl.circular_projection` stacked with pseudotime, which
    collapses ``n-1`` dimensions onto two.

    The tree is handed back exactly as the fit left it; nothing is removed from it. Alongside
    what :mod:`scFates` records, it carries the keys the projection reads and writes, which
    belong to :mod:`cellrank`:

    - ``.obsm['lineages_fwd']``, the fate probabilities the tree was fitted from. A plain
      array.
    - ``.obsm['X_fate_simplex_fwd']``, the circular projection, which is ``.obsm['X_fates']``
      without its pseudotime column.
    - ``.obs['term_states_fwd']`` and ``.uns['term_states_fwd_colors']``, naming and colouring
      the fates for the projection.
    - ``.obs['lineages_fwd_kl_divergence']``, which restates ``.obs['kl_divergence']``.

    Examples
    --------
    >>> atlas.pl.fate_tree(mudata, color="celltype")
    >>> fitted = atlas.pl.fate_tree(mudata, return_tree=True)
    >>> atlas.pl.fate_tree(mudata, tree=fitted, color="leiden")
    """
    scf = _require_scfates("`atlas.pl.fate_tree`")

    _resolve_basis(mudata, basis)
    probabilities = _fate_frame(mudata, fate_probability_key).loc[mudata.obs_names]

    if probabilities.shape[1] < 2:
        raise ValueError(
            f"mudata.obsm['{fate_probability_key}'] records {probabilities.shape[1]} fate(s); a "
            f"branching structure needs at least two fates to run between"
        )

    _reject_unfittable(probabilities, fate_probability_key)

    if time_key not in mudata.obs.columns:
        raise KeyError(
            f"'{time_key}' not in mudata.obs; run {_TIME_PRODUCER} to compute a pseudotime, "
            f"or pass `time_key` naming the column holding one"
        )

    if color is not None and color not in mudata.obs.columns:
        raise KeyError(f"'{color}' not in mudata.obs, which is what `color` names for this figure")

    if tree is None:
        tree = _fit_tree(
            mudata,
            probabilities,
            basis,
            time_key,
            root,
            nodes,
            method,
            ppt_lambda,
            random_state,
            dict(tree_kwargs or {}),
        )
    else:
        if not tree.obs_names.equals(mudata.obs_names):
            raise ValueError(
                f"`tree` was fitted on {tree.n_obs} cell(s) and mudata holds {mudata.n_obs}, "
                f"which are not the same cells; refit the tree on this object rather than "
                f"drawing cells it no longer holds"
            )
        if color is not None and color not in _TREE_COLUMNS:
            tree.obs[color] = mudata.obs[color]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (12, 5))
        axes = list(axes)
    else:
        axes = list(ax)
        if len(axes) != 2:
            raise ValueError(
                f"`ax` takes the two axes to draw into — the embedding, then the dendrogram — and {len(axes)} was given"
            )
        fig = axes[0].figure

    graph_kwargs, dendrogram_kwargs = dict(graph_kwargs or {}), dict(dendrogram_kwargs or {})
    categorical = color is not None and isinstance(mudata.obs[color].dtype, pd.CategoricalDtype)

    graph_kwargs.setdefault("color_cells", color)
    dendrogram_kwargs.setdefault("color", color)

    graph_kwargs.setdefault("legend_loc", "none" if categorical else None)
    graph_kwargs.setdefault("colorbar_loc", None)

    scf.pl.graph(tree, basis=basis, ax=axes[0], show=False, **graph_kwargs)
    scf.pl.dendrogram(tree, ax=axes[1], show=False, **dendrogram_kwargs)

    if categorical:
        handles, labels = axes[1].get_legend_handles_labels()
        if axes[1].get_legend() is not None:
            axes[1].get_legend().remove()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()

    if save is not None:
        figure_path = os.path.join(os.getcwd(), "figures")
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(os.path.join(figure_path, f"fate_tree_{save}.png"), bbox_inches="tight", dpi=300)

    if show is None:
        show = not return_tree
    if show:
        plt.show()

    return tree if return_tree else axes
