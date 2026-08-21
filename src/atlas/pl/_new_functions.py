import os
import warnings
from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from muon import MuData

from atlas.tl import MultiLineageGAM

from .utils import _fate_frame, _resolve_basis, _resolve_color, _state_colors

_DEFAULT_LINEAGE_COLOR = "grey"
_UNDECIDED_COLOR = "lightgrey"


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
    selects the feature instead, and is also what distinguishes a feature name carried by
    more than one modality.

    Colouring by a categorical key records the colours chosen for it on ``mudata`` as
    ``.uns["<key>_colors"]``, following the convention the scverse ecosystem reads. The
    values therefore keep their colours across calls, and agree with figures drawn by
    :func:`scanpy.pl.embedding` or :func:`muon.pl.embedding` directly. Colours already
    recorded there are used as they are. This annotates the object that is passed.

    No colour map is imposed. When ``cmap`` is not given, the matplotlib default applies, so
    a colour map set globally through :func:`scanpy.set_figure_params` is respected.

    ``save`` writes to ``figures/<basis><save>``, as in :func:`scanpy.pl.embedding`.

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
        Filename to save the figure under, in ``figures/``. Honoured whether or not anything
        is returned.

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
        Filename to save to, as in :func:`scvelo.pl.scatter`.
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

    Notes
    -----
    Many fates crowd the figure: the number of diverging maps grows as ``n(n-1)/2``, and one
    on-data label per fate can overlap in a dense embedding.

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
