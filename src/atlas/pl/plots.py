import os
import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from muon import MuData

from .utils import MultiBranchGAM, _require_scfates


def plot_embedding(
    mudata: MuData,
    embedding_key: str = "X_umap",
    observation: str = "pseudotime",
    save: bool | str | None = None,
    cmap: str = "Blues",
    **kwargs,
) -> None:
    """
    Plot a low-dimensional embedding of cells colored by a given observation.

    This function creates a scatter plot of cells in a specified embedding space
    (e.g. UMAP, PCA) using values from `mudata.obs` for coloring.

    .. deprecated:: 1.1.0
       This function is deprecated and will be removed in version 2.0.0.
       It is retained in version 1.1.0 for backwards compatibility.

       It is superseded by :func:`~atlas.pl.embedding`.

    Parameters
    ----------
    mudata
        MuData object containing cell-level metadata in `.obs` and embeddings in `.obsm`.
    embedding_key
        Key in `mudata.obsm` specifying the embedding to use for visualization
        (e.g. ``"X_umap"``, ``"X_pca"``).
    observation
        Column name in `mudata.obs` used to color the cells.
    save
        If ``True``, save the figure using scvelo defaults. If a string, specifies
        the filename. If ``None``, the plot is not saved.
    cmap
        Colormap used for continuous observations. Ignored for categorical variables.
    **kwargs
        Additional keyword arguments passed to :func:`scvelo.pl.scatter`.

    Returns
    -------
    None
        Displays (and optionally saves) the embedding plot.

    Warns
    -----
    UserWarning
        If `observation` is not found in `mudata.obs`.

    UserWarning
        If `embedding_key` is not found in `mudata.obsm`.
    """
    warnings.warn(
        "`plot_embedding` is deprecated since version 1.1.0 and will be "
        "removed in version 2.0.0. Use `atlas.pl.embedding` instead.",
        FutureWarning,
        stacklevel=2,
    )

    import scvelo as scv

    if observation not in mudata.obs.columns:
        warnings.warn(f"WARNING: {observation} not a valid cell metadata", stacklevel=2)
        return

    if embedding_key not in mudata.obsm:
        warnings.warn(f"WARNING: {embedding_key} not a valid cell embedding", stacklevel=2)
        return

    _tmp = AnnData(X=np.zeros((mudata.n_obs, 1)), obs=mudata.obs.copy())
    _tmp.obsm[embedding_key] = mudata.obsm[embedding_key]
    title = observation
    kwargs.setdefault("save", save)

    scv.pl.scatter(_tmp, title=title, color=observation, color_map=cmap, **kwargs)


def plot_fate_probabilities(
    mudata: MuData,
    embedding_key: str = "X_umap",
    fate_probability_key: str = "fate_probabilities",
    states: str | Sequence[str] | None = None,
    cmap: str = "viridis",
    title: str = "Fate Probabilities",
    save: bool | str | None = None,
    **kwargs,
) -> None:
    # This function is adapted from CellRank (BSD 3-Clause License).
    # Original source: https://github.com/scverse/cellrank
    # Copyright (c) 2019, Theis Lab
    # Modifications:
    # - Constructs Lineage class from MuData object
    # - Only focuses on plotting fate probabilities and not time.
    # - Uses original scvelo plot function with custom cmap to plot fate probabilities
    # - Controls for singleton and nans as in the original CellRank framework
    """
    Plot lineage fate probabilities on a low-dimensional embedding.

    This function visualizes fate probabilities similarly to :cite:`cellrank1`, by projecting
    them onto a specified embedding (such as UMAP). Cells are colored using
    lineage-specific gradients, allowing inspection of differentiation trajectories.

    .. deprecated:: 1.1.0
       This function is deprecated and will be removed in version 2.0.0.
       It is retained in version 1.1.0 for backwards compatibility.

       It is superseded by :func:`~atlas.pl.fate_probabilities`

    Parameters
    ----------
    mudata
        MuData object.
    embedding_key
        Key in `mudata.obsm` specifying the embedding to use for visualization
        (e.g. ``"X_umap"``, ``"X_pca"``).
    fate_probability_key
        Key in `mudata.obsm` containing a DataFrame of fate probabilities.
    states
        Subset of terminal states (lineages) to visualize. Can be a single string
        or a sequence of strings. If ``None``, all available states are plotted.
    cmap
        Colormap used for visualizing fate probabilities.
    title
        Title of the plot.
    save
        If ``True``, save the figure using scvelo defaults. If a string, specifies
        the filename. If ``None``, the plot is not saved.
    **kwargs
        Additional keyword arguments passed to :func:`scvelo.pl.scatter`.

    Warns
    -----
    UserWarning
        If `fate_probability_key` is not found in `mudata.obsm`.
    UserWarning
        If `embedding_key` is not found in `mudata.obsm`.
    UserWarning
        If no valid terminal states are selected.

    """
    warnings.warn(
        "`plot_fate_probabilities` is deprecated since version 1.1.0 and will be "
        "removed in version 2.0.0. Use `atlas.pl.fate_probabilities` instead.",
        FutureWarning,
        stacklevel=2,
    )

    import scvelo as scv
    from cellrank._utils._lineage import Lineage

    if fate_probability_key not in mudata.obsm.keys():
        warnings.warn("WARNING: fate probabilities are not available; Try recompute them.", stacklevel=2)
        return

    if embedding_key not in mudata.obsm:
        warnings.warn("WARNING: embedding is not available.", stacklevel=2)
        return

    fate_probabilities = mudata.obsm[fate_probability_key]
    _terminal_states = list(fate_probabilities.columns)
    _all_colors = mudata.uns["fate_state_colors"]

    if states is not None and isinstance(states, str):
        states = [states]
    states = [s for s in states if s in _terminal_states] if states is not None else _terminal_states
    if not len(states):
        warnings.warn("WARNING: No lineages have been selected.", stacklevel=2)
        return

    # compatibility with scvelo plot
    _terminal_colors = [_all_colors[state] for state in _terminal_states]
    _data = Lineage(fate_probabilities.to_numpy(), names=_terminal_states, colors=_terminal_colors)

    _data = _data[states].copy()
    _singleton = _data.shape[1] == 1
    _X = _data.X

    _is_singleton_all_one = _singleton and np.allclose(_X, 1.0)

    if not _is_singleton_all_one and not _singleton:
        for col in _X.T:
            finite = np.isfinite(col)
            _has_intermediate = np.any(finite & (col > 0.0) & (col < 1.0))
            if _has_intermediate:
                mask = ~np.isclose(col, 1.0)
                if np.any(mask):
                    col[~mask] = np.nanmax(col[mask])

    _tmp = AnnData(X=np.zeros((mudata.n_obs, 1)), obs=mudata.obs.copy())
    _tmp.obsm[embedding_key] = mudata.obsm[embedding_key]

    kwargs.setdefault("save", save)
    kwargs.setdefault("legend_loc", "on data")

    if _is_singleton_all_one:
        state = states[0]
        color_key = f"_fate_singleton_{state}"
        _tmp.obs[color_key] = _all_colors[state]
        _ = kwargs.pop("color_map", None)
        _ = kwargs.pop("color_gradients", None)
        _ = kwargs.pop("cmap", None)
        _ = kwargs.pop("perc", None)
        kwargs["colorbar"] = False
        kwargs["color"] = color_key
        kwargs["palette"] = [_all_colors[state]]
    else:
        kwargs["color_gradients"] = _data

    scv.pl.scatter(_tmp, title=title, basis=embedding_key.replace("X_", ""), **kwargs)


def plot_trends(
    mudata: MuData,
    ptf: str,
    gene: str,
    time_key: str = "pseudotime",
    fate_probability_key: str = "fate_probabilities",
    branches: list | str | None = None,
    sharex: bool = False,
    n_splines: int = 8,
    n_points: int = 200,
    order: int = 1,
    save: str | None = None,
) -> None:
    """
    Plot branch-specific dynamics of transcription factor expression and gene activity.

    This function fits branch-specific Generalized Additive Models (GAMs) using
    :class:`pygam.pygam.LinearGAM` and visualizes smooth trends along pseudotime for a
    transcription factor (TF) and a target gene. Each lineage (branch) is modeled
    independently using fate probabilities as weights.

    For each branch, the function displays the fitted GAM curve together with its
    confidence interval.

    .. deprecated:: 1.1.0
       This function is deprecated and will be removed in version 2.0.0.
       It is retained in version 1.1.0 for backwards compatibility.

       It is superseded by :func:`~atlas.pl.trends`.

    Parameters
    ----------
    mudata
        MuData object containing gene expression and gene activity modalities.
    ptf
        Name of the transcription factor in ``mudata["rna"].var_names``.
    gene
        Name of the gene in ``mudata["activity"].var_names``.
    time_key
        Key in ``mudata.obs`` containing pseudotime values.
    fate_probability_key
        Key in ``mudata.obsm`` containing fate probabilities for each branch.
    branches
        Subset of branches to plot. If ``None``, all available branches are shown.
    sharex
        Whether to share the x-axis (pseudotime) between the two subplots.
    n_splines
        Number of spline basis functions used in the GAM.
    n_points
        Number of points used to evaluate the fitted curves.
    order
        Currently unused parameter reserved for future extensions.
    save
        If provided, saves the figure to ``figures/trends_<save>.png`` in the
        current working directory.

    Returns
    -------
    None
        The function generates a matplotlib figure and optionally saves it.

    """
    warnings.warn(
        "`plot_trends` is deprecated since version 1.1.0 and will be "
        "removed in version 2.0.0. Use `atlas.pl.trends` instead.",
        FutureWarning,
        stacklevel=2,
    )

    if ptf not in mudata["rna"].var_names:
        raise KeyError(f"TF {ptf} not available")
    if gene not in mudata["activity"].var_names:
        raise KeyError(f"Gene {gene} not available")
    if time_key not in mudata.obs:
        raise KeyError(f"{time_key} not in mudata.obs")
    if fate_probability_key not in mudata.obsm:
        raise KeyError(f"{fate_probability_key} not in mudata.obsm")

    mbgam = MultiBranchGAM(
        mudata=mudata,
        ptf=ptf,
        gene=gene,
        time_key=time_key,
        fate_prob_key=fate_probability_key,
        n_splines=n_splines,
    )
    mbgam.fit()
    mbgam.predict(n_points=n_points)
    predictions = mbgam.predictions

    if branches is None:
        branches = list(predictions.keys())
    elif isinstance(branches, str):
        branches = [branches]

    colors = mudata.uns.get("fate_state_colors", {})
    default_color = "grey"

    fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=sharex)

    for branch in branches:
        pred = predictions.get(branch, None)
        if pred is None:
            continue
        t = pred["t_grid"]
        c = colors.get(branch, default_color)

        axes[0].plot(t, pred["gex"], color=c, lw=2, label=branch)
        axes[0].fill_between(t, pred["gex_lower"], pred["gex_upper"], color=c, alpha=0.2)
        axes[1].plot(t, pred["act"], color=c, lw=2, label=branch)
        axes[1].fill_between(t, pred["act_lower"], pred["act_upper"], color=c, alpha=0.2)

    axes[0].set_ylabel(f"{ptf} expression (z-score)")
    axes[1].set_ylabel(f"{gene} activity (z-score)")
    axes[1].set_xlabel(time_key)
    axes[0].set_title(f"Dynamics across lineages: {ptf} -> {gene}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save:
        figure_path = os.path.join(os.getcwd(), "figures")
        if not os.path.exists(figure_path):
            os.mkdir(figure_path)
        path = os.path.join(figure_path, f"trends_{save}.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)


def plot_tree(
    mudata: MuData,
    embedding_key: str = "umap",
    fate_probability_key: str = "fate_probabilities",
    time_key: str = "pseudotime",
    nodes: int = 300,
    method: Literal["ppt", "epg"] = "ppt",
    ppt_lambda: int = 100,
    auto_root: bool = False,
    root_params: dict = None,
    reassign_pseudotime: bool = False,
    crowdedness: float = 1,
    color: str | None = None,
    color_milestones: bool = False,
    n_jobs: int = -1,
    n_map: int = 1,
    save: str | None = None,
    random_state: int = 42,
    **kwargs,
) -> None:
    # This function adapts a tutorial from scFates (BSD 3-Clause).
    # Original source: https://scfates.readthedocs.io/en/latest/Conversion_from_CellRank_pipeline.html
    # Copyright (c) 2020, LouisFaure
    # Modifications:
    # - Adapted the code to MuData object resulting from ATLAS
    # - Constructs instance of the Lineage class from CellRank to run tree inference
    """
    Compute and visualize a principal tree from fate probabilities, using :cite:`scfates`.

    .. deprecated:: 1.1.0
       This function is deprecated and will be removed in version 2.0.0.
       It is retained in version 1.1.0 for backwards compatibility.

       It is superseded by :func:`~atlas.pl.fate_tree`.

    Parameters
    ----------
    mudata
        Annotated multi-modal data object.
    embedding_key
        Key in ``mudata.obsm`` for the embedding (expects ``"X_{embedding_key}"``).
    fate_probability_key
        Key in ``mudata.obsm`` where fate probabilities are stored.
    nodes
        Number of nodes used to fit the principal tree.
    method
        Tree inference method, one of ``{'ppt', 'epg'}``.
    ppt_lambda
        Regularization parameter for ``'ppt'`` method.
    auto_root
        Whether to automatically infer the root.
    root_params
        Additional keyword arguments for root inference.
    reassign_pseudotime
        Whether to recompute pseudotime after tree construction.
    crowdedness
        Controls dendrogram layout spacing.
    color
        Key in ``mudata.obs`` used for coloring cells.
    color_milestones
        Whether to color milestones in the dendrogram.
    n_jobs
        Number of parallel jobs for pseudotime computation.
    n_map
        Number of mappings used in pseudotime computation.
    save
        Filename or path to save the figures.
    random_state
        Random seed.
    **kwargs
        Additional arguments passed to :func:`scFates.tl.cellrank_to_tree`
        and plotting functions.

    Warns
    -----
    UserWarning
        If keys are not correctly specified by the user.
    UserWarning
        If no terminal states are found or a single terminal state is recovered.

    Notes
    -----
    Requires precomputed fate probabilities with at least 2 terminal states.
    """
    warnings.warn(
        "`plot_tree` is deprecated since version 1.1.0 and will be "
        "removed in version 2.0.0. Use `atlas.pl.fate_tree` instead.",
        FutureWarning,
        stacklevel=2,
    )

    scf = _require_scfates("`atlas.pl.plot_tree`")
    from cellrank._utils._lineage import Lineage

    if root_params is None:
        root_params = {}
    fate_prob_key, lineage_key = "term_states_fwd_memberships", "lineages_fwd"

    if fate_probability_key not in mudata.obsm.keys():
        warnings.warn("WARNING: fate probabilities are not available; Try recompute them", stacklevel=2)
        return

    if "kl_divergence" not in mudata.obs:
        warnings.warn("Entropy as KL-divergence required; Recompute entropy.", stacklevel=2)
        return

    if f"X_{embedding_key}" not in mudata.obsm:
        warnings.warn(f"X_{embedding_key} not in mudata.obsm", stacklevel=2)
        return

    if color is not None and color not in mudata.obs.columns:
        warnings.warn(f"Specified color ({color}) not in mudata.obs", stacklevel=2)
        return

    fate_probabilities = mudata.obsm[fate_probability_key]
    if fate_probabilities is None:
        warnings.warn("WARING: fate probabilties are not available", stacklevel=2)
        return
    fate_probabilities = fate_probabilities.loc[mudata.obs_names]

    # scFates.cellrank_to_tree does not check n_fates = 1 and cellrank.pl.circular_projection does not work.
    if fate_probabilities.shape[1] == 0:
        warnings.warn("Fate probabilities are not available; Try recompute them", stacklevel=2)
        return

    if fate_probabilities.shape[1] < 2:
        warnings.warn("WARNING: only one fate has been found, principal tree not available", stacklevel=2)
        return

    _terminal_states = list(fate_probabilities.columns)
    _all_colors = mudata.uns["fate_state_colors"]
    _terminal_colors = [_all_colors[s] for s in _terminal_states]
    _data = Lineage(fate_probabilities.to_numpy(), names=_terminal_states, colors=_terminal_colors)

    _tmp = AnnData(X=np.zeros((mudata.n_obs, 1)), obs=mudata.obs.copy())
    _tmp.obsm[fate_prob_key] = mudata.obsm[fate_probability_key].values
    _tmp.obsm[lineage_key] = _data
    _tmp.obsm[f"X_{embedding_key}"] = mudata.obsm[f"X_{embedding_key}"]

    scf.tl.cellrank_to_tree(
        adata=_tmp,
        time=time_key,
        Nodes=nodes,
        method=method,
        ppt_lambda=ppt_lambda,
        auto_root=auto_root,
        root_params=root_params,
        reassign_pseudotime=reassign_pseudotime,
        key_cellrank=fate_prob_key,
        copy=False,
        **kwargs,
    )

    initial_cells = next(iter(mudata.uns["initial_states"].values()))
    initial_idx = _tmp.obs_names.get_indexer(initial_cells)
    R_init = _tmp.obsm["X_R"][initial_idx, :]
    root = int(R_init.mean(axis=0).argmax())

    scf.tl.root(_tmp, root)
    scf.tl.pseudotime(_tmp, n_jobs=n_jobs, n_map=n_map, seed=random_state, copy=False)
    scf.tl.dendrogram(_tmp, crowdedness=crowdedness)
    plt.close()  # scf.tl.dendrogram opens a ulterior plot which we don't want to see.

    scf.pl.graph(_tmp, basis=embedding_key, save=save, **kwargs)
    scf.pl.dendrogram(_tmp, color_milestones=color_milestones, color=color, save=save, **kwargs)
