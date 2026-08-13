import warnings
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib.colors import to_hex
from muon import MuData
from scipy.sparse import csr_matrix

#: Version in which the parameters superseded by ``key`` stop working.
_KEY_REMOVAL_VERSION = "2.0.0"

#: The kinds of state recorded as categorical columns of ``mudata.obs``. Separate columns are
#: what lets a cell belong to states of more than one kind, which ``allow_overlap`` permits.
_STATE_KINDS = ("initial_states", "terminal_states", "macrostates")

#: ATLAS's record of the colour assigned to each state name. Deliberately not named
#: ``*_colors``: the ecosystem reads such a key positionally for a column of the same stem,
#: which would discard a mapping. Nothing reads colours from here — the per-kind lists are
#: what a consumer reads — but the assignment survives here when those are overwritten.
_PALETTE_KEY = "atlas_state_palette"

#: Superseded colour mapping, retained so `atlas.pl` keeps working, removed in 2.0.0.
_LEGACY_PALETTE_KEY = "fate_state_colors"


def _deprecated_key_arg(old_name: str, value, key, key_value) -> None:
    """Warn that ``old_name`` is superseded, or fail if it is combined with ``key``.

    Parameters
    ----------
    old_name
        Name of the superseded parameter, as the caller wrote it.
    value
        What the caller passed for it, or ``None`` if they did not.
    key
        Name of the parameter replacing it.
    key_value
        What the caller passed for ``key``, or ``None`` if they did not.

    Raises
    ------
    ValueError
        If both are supplied. Which was intended cannot be known, and preferring either
        would silently select a graph the caller did not ask for.
    """
    if value is None:
        return

    if key_value is not None:
        raise ValueError(
            f"`{old_name}` and `{key}` were both supplied; pass only `{key}`, "
            f"since which graph was intended cannot be determined from both."
        )

    warnings.warn(
        f"`{old_name}` is superseded by `{key}` and will be removed in {_KEY_REMOVAL_VERSION}. "
        f"Pass `{key}` naming the graph's record instead; its matrices are resolved from it.",
        FutureWarning,
        stacklevel=3,
    )


def _resolve_graph_key(mudata: MuData, key: str, kind: str) -> str:
    """Resolve the name of a stored graph matrix from the record describing it.

    The record written alongside a neighbor graph names each matrix in full, so the name is
    looked up rather than assembled: the ``{key}_{kind}`` pattern is a convention of
    whoever wrote the graph, not a guarantee.

    Falls back to that convention when the record is absent or omits the name, so that
    objects carrying a graph without a complete record keep working.

    Parameters
    ----------
    mudata
        Multimodal annotated data object.
    key
        Key in ``mudata.uns`` under which the graph is recorded.
    kind
        Either ``"distances"`` or ``"connectivities"``.

    Returns
    -------
    Name of the matrix in ``mudata.obsp``.

    Raises
    ------
    KeyError
        If neither the record nor the conventional name identifies a matrix that is
        present.
    """
    record = mudata.uns.get(key)
    name = record.get(f"{kind}_key") if isinstance(record, Mapping) else None
    if name is None:
        name = f"{key}_{kind}"

    if name not in mudata.obsp:
        raise KeyError(f"could not resolve the {kind} matrix for '{key}': '{name}' not in mudata.obsp")
    return name


def _cellrank_anndata(mudata: MuData, connectivity_key: str, cluster_key: str | None = None) -> AnnData:
    """Build the temporary :class:`~anndata.AnnData` required by the CellRank interface.

    CellRank inspects the shape and variable names of the object it is given, so this
    view carries a correctly shaped placeholder matrix, the variable index of the
    ``"rna"`` modality, the observation frame, and the requested pairwise matrix.

    The original ``MuData`` is left unchanged.

    Parameters
    ----------
    mudata
        Multimodal annotated data object.
    connectivity_key
        Key in ``mudata.obsp`` holding the connectivity matrix to carry over.
    cluster_key
        Column of ``mudata.obs`` to cast to ``category``, as CellRank expects. Ignored
        when ``None``.

    Returns
    -------
    A temporary :class:`~anndata.AnnData` view over ``mudata``.
    """
    adata = AnnData(
        X=csr_matrix((mudata.n_obs, mudata["rna"].n_vars)),
        obs=mudata.obs.copy(),
        var=pd.DataFrame([], index=mudata["rna"].var_names),
    )
    adata.obsp[connectivity_key] = mudata.obsp[connectivity_key]

    if cluster_key is not None:
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    return adata


def _palantir_anndata(mudata: MuData, eigvec_keys: Sequence[str], multiscale: pd.DataFrame | np.ndarray) -> AnnData:
    """Build the temporary :class:`~anndata.AnnData` required by the Palantir interface.

    Palantir's cell-selection helpers read only ``obs``, ``obs_names`` and one
    multidimensional annotation, so this view carries nothing else.

    The multiscale representation is attached under every key in ``eigvec_keys``, all
    referencing the same data. This exists because Palantir versions below 1.4.5 do not
    forward the ``eigvec_key`` argument of :func:`palantir.utils.early_cell` to
    :func:`palantir.utils.fallback_terminal_cell`, which therefore looks the
    representation up under Palantir's own default name and fails when the caller stored
    it elsewhere. Registering both names lets the fallback operate on the same
    representation ATLAS uses, on every supported Palantir version.

    The original ``MuData`` is left unchanged.

    Parameters
    ----------
    mudata
        Multimodal annotated data object.
    eigvec_keys
        Keys under which to register ``multiscale`` in ``.obsm``.
    multiscale
        Multiscale diffusion representation, of shape ``(n_cells, n_components)``.

    Returns
    -------
    A temporary :class:`~anndata.AnnData` view over ``mudata``.
    """
    adata = AnnData(obs=mudata.obs.copy())
    for key in eigvec_keys:
        adata.obsm[key] = multiscale
    return adata


def _states_to_column(states: Mapping[str, Sequence[str]], obs_names: pd.Index) -> pd.Series:
    """Turn a mapping of state name to cell names into a categorical over every cell.

    The inverse of :func:`_invert_assignment`, which existed because ATLAS converted the
    categorical CellRank hands it into a dictionary. `PalantirExtension` never has a
    categorical to keep, and neither does an object written under the superseded layout, so
    both need the conversion in this direction.

    Cells belonging to no state carry no value. Categories are the state names given, so a
    label of the annotation states were named after that no state corresponds to does not
    appear.
    """
    assignment = pd.Series(pd.NA, index=obs_names, dtype=object)

    for name, cells in states.items():
        assignment.loc[list(cells)] = str(name)

    return assignment.astype(pd.CategoricalDtype(categories=[str(name) for name in states]))


def _resolve_overlap(initial: pd.Series, terminal: pd.Series) -> pd.Series:
    """Combine the two kinds into the single-valued view ``macrostates`` needs.

    A cell may carry both assignments — ``allow_overlap`` exists to permit it — and a
    categorical holds one value per cell, so one has to give way. The initial assignment
    takes precedence, following CellRank, and the caller is warned.

    In practice the two agree: a cell belongs to at most one macrostate, and initial and
    terminal cells are drawn from macrostates, so an overlap means one macrostate was
    designated both and carries one name. The precedence keeps the result well defined rather
    than changing any value.
    """
    overlapping = initial.notna() & terminal.notna()

    if overlapping.any():
        cells = list(initial.index[overlapping])
        shown = ", ".join(cells[:5]) + (f", … ({len(cells)} in total)" if len(cells) > 5 else "")
        warnings.warn(
            f"{len(cells)} cell(s) belong to both an initial and a terminal state ({shown}); "
            f"`macrostates` records the initial assignment for them. Both remain in "
            f"`initial_states` and `terminal_states`.",
            stacklevel=3,
        )

    combined = terminal.astype(object).where(terminal.notna(), other=None)
    combined = combined.where(~initial.notna(), other=initial.astype(object))
    names = [str(name) for name in dict.fromkeys(list(terminal.cat.categories) + list(initial.cat.categories))]

    return combined.astype(pd.CategoricalDtype(categories=names))


def _intermediate_states(mudata: MuData) -> dict[str, list[str]]:
    """The recorded states that are neither where a trajectory begins nor where it ends.

    Derived rather than stored: a fourth column would be a fourth colour list to keep in step,
    and would go stale whenever any of the three it reads changed, for no information not
    already present in them.

    Its durable consumer is evaluation. `atlas.tl.evaluate` does not read intermediate states
    yet — wiring it up belongs to a later change — so today this only feeds the superseded
    ``uns["intermediate_states"]``, and outlives it.

    Where ``macrostates`` is the union of the other kinds rather than a coarse-graining, which
    is every path but CellRank inferring its own states, nothing can lie between them and the
    result is empty. That is a true answer, not an inability to answer.

    Raises
    ------
    KeyError
        If no ``macrostates`` column is recorded, there being nothing to derive from.
    """
    macrostates = mudata.obs.get("macrostates")
    if macrostates is None:
        raise KeyError(
            "`mudata.obs['macrostates']` is absent, so the intermediate states cannot be "
            "determined; run a trajectory inference, or `atlas.tl.migrate_states` for an "
            "object written before this layout."
        )

    initial, terminal = mudata.obs.get("initial_states"), mudata.obs.get("terminal_states")
    designated = pd.Series(False, index=mudata.obs_names)
    for column in (initial, terminal):
        if column is not None:
            designated |= column.notna().to_numpy()

    remaining = macrostates[macrostates.notna() & ~designated]
    if isinstance(remaining.dtype, pd.CategoricalDtype):
        remaining = remaining.cat.remove_unused_categories()

    return _invert_assignment(remaining)


def _invert_assignment(assignment):
    if not isinstance(assignment.dtype, pd.CategoricalDtype):
        assignment = assignment.astype("category")

    inverted_assignment = {state: assignment.index[assignment == state].tolist() for state in assignment.cat.categories}
    return inverted_assignment


def _state_names(mudata: MuData) -> set[str]:
    """Every state name the object records, across all kinds.

    Colours are assigned over this union rather than per kind. Assigning per kind would draw
    each kind's palette from the start of the colour cycle, giving unrelated states the same
    colour and one state appearing under two kinds two different colours.
    """
    names: set[str] = set()

    for kind in _STATE_KINDS:
        column = mudata.obs.get(kind)
        if column is not None and isinstance(column.dtype, pd.CategoricalDtype):
            names.update(str(name) for name in column.cat.categories)

    # The superseded dictionaries are still written, and carry `intermediate_states`, which is
    # not a kind of its own. Reading them keeps the palette complete while they exist.
    for key in ("terminal_states", "initial_states", "intermediate_states"):
        states = mudata.uns.get(key)
        if isinstance(states, Mapping):
            names.update(str(name) for name in states)

    return names


def _guard_palette_collision(mudata: MuData) -> None:
    """Refuse to write colours when an annotation would consume the superseded mapping.

    ``uns[_LEGACY_PALETTE_KEY]`` is a mapping under a name the convention reads positionally
    for a matching column. Where that column exists, scanpy takes the mapping's keys for
    colour values, discards it and overwrites it with a list of its own. ``atlas.pl`` still
    reads that mapping, so losing it breaks plotting.

    Raises
    ------
    ValueError
        If an ``obs`` column would cause the superseded mapping to be read as its colours.
    """
    colliding = _LEGACY_PALETTE_KEY.removesuffix("_colors")
    if colliding in mudata.obs.columns:
        raise ValueError(
            f"`mudata.obs['{colliding}']` collides with `mudata.uns['{_LEGACY_PALETTE_KEY}']`, "
            f"which is a mapping: plotting that column would read the mapping positionally and "
            f"overwrite it, and `atlas.pl` still reads it. Rename the column."
        )


def _assign_state_colors(mudata: MuData, cmap: str = "tab20") -> None:
    """Give every state name a colour, and record that assignment everywhere it is read.

    Assignment is over the union of the names of all kinds, taken in sorted order, so it
    depends on the names alone and not on which inference produced them. A name already
    carrying a colour keeps it, so repeating an inference does not recolour what it finds
    again.

    The assignment is written to ``uns[_PALETTE_KEY]``, to the conventional per-kind colour
    lists, and to the superseded mapping, so none of them can disagree.
    """
    _guard_palette_collision(mudata)

    palette = mudata.uns.setdefault(_PALETTE_KEY, {})
    unassigned = sorted(name for name in _state_names(mudata) if name not in palette)

    if unassigned:
        base_colors = [to_hex(color) for color in plt.get_cmap(cmap).colors]
        taken = set(palette.values())
        # Prefer a colour nothing already uses, so distinct states stay distinguishable;
        # fall back to cycling once the map is exhausted.
        free = [color for color in base_colors if color not in taken]
        for offset, name in enumerate(unassigned):
            palette[name] = free[offset] if offset < len(free) else base_colors[offset % len(base_colors)]

    _write_state_colors(mudata)


def _write_state_colors(mudata: MuData) -> None:
    """Write the recorded assignment into every place a colour is read from.

    Each kind's list is built by looking its categories up in the record, never by pairing two
    orderings: the record is ordered by name and the categories are not, so zipping them would
    misalign every list while still producing one of the right length.
    """
    palette = mudata.uns.get(_PALETTE_KEY, {})

    for kind in _STATE_KINDS:
        column = mudata.obs.get(kind)
        if column is None or not isinstance(column.dtype, pd.CategoricalDtype):
            continue
        categories = [str(name) for name in column.cat.categories]
        if all(name in palette for name in categories):
            mudata.uns[f"{kind}_colors"] = [palette[name] for name in categories]

    mudata.uns[_LEGACY_PALETTE_KEY] = dict(palette)


def reset_state_colors(mudata: MuData) -> None:
    """Restore the state colours ATLAS assigned.

    The per-kind colour lists are the keys the wider ecosystem reads, which means it also
    writes them: passing ``palette`` to :func:`scanpy.pl.embedding` for one state column
    replaces that list for good, so a state appearing under more than one kind can end up
    differently coloured under each.

    This rewrites every list from ``mudata.uns["atlas_state_palette"]``, the record of what was
    assigned. It restores rather than reconciles, so it needs no rule for preferring one kind
    over another and works however many lists were overwritten, up to all of them.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying states recorded by ATLAS.

    Returns
    -------
    None
        The colour lists are rewritten in place.

    Raises
    ------
    KeyError
        If the object carries no record of an assignment, there being nothing to restore.

    Examples
    --------
    >>> import scanpy as sc
    >>> sc.pl.embedding(mudata, basis="umap", color="terminal_states", palette="tab10")
    >>> atlas.tl.reset_state_colors(mudata)
    """
    if not mudata.uns.get(_PALETTE_KEY):
        raise KeyError(
            f"`mudata.uns['{_PALETTE_KEY}']` is absent or empty, so there is no recorded "
            f"assignment to restore; run a trajectory inference, or `atlas.tl.migrate_states` "
            f"for an object written before this layout."
        )

    _write_state_colors(mudata)


def _minmax(x: np.ndarray) -> np.ndarray:
    xmax, xmin = np.max(x), np.min(x)
    if xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def compute_entropy(mudata: MuData, fate_probability_key: str = "fate_probabilities") -> None:
    # This function is adapted from CellRank (BSD 3-Clause License).
    # Original source: https://github.com/scverse/cellrank
    # Copyright (c) 2019, Theis Lab
    # Modifications:
    # - Directly works on MuData object.
    # - Separate funcion for min-max scaling.
    # - Added input validation according to ATLAS requirements.
    # - Removed early_cells subset and works on the whole cell set.
    """Compute entropy-based metrics from fate probabilities.

    This function computes per-cell entropy measures from fate probability
    distributions stored in ``mudata.obsm``. Specifically, it calculates:

    - Shannon entropy, measuring the uncertainty of cell fate assignment.
    - Kullback-Leibler (KL) divergence from the average fate distribution.

    Both metrics are min-max scaled to the [0, 1] interval and stored in
    ``mudata.obs``.

    Parameters
    ----------
    mudata
        MuData object containing fate probabilities in ``.obsm``.
    fate_probability_key
        Key in ``mudata.obsm`` where fate probabilities are stored as a
        :class:`pandas.DataFrame`. Rows correspond to cells and columns
        to terminal states.

    Returns
    -------
    None
        The input ``mudata`` object is updated in place with two new columns
        in ``.obs``:

        - ``"shannon_entropy"``
        - ``"kl_divergence"``

    Raises
    ------
    ValueError
        If fate probabilities are not present in ``mudata.obsm``.
    ValueError
        If the stored fate probabilities are not a pandas DataFrame.

    Warns
    -----
    UserWarning
        If no terminal states are present (i.e., zero columns in the
        probability matrix). In this case, entropy values are set to NaN.

    Examples
    --------
    >>> compute_entropy(mdata)
    >>> mdata.obs["shannon_entropy"].head()
    >>> mdata.obs["kl_divergence"].head()
    """
    probs = mudata.obsm.get(fate_probability_key, None)
    if probs is None:
        raise ValueError("Fate probabilities are not available. Try recompute them.")
    if not isinstance(probs, pd.DataFrame):
        raise ValueError("Fate probabilities must be a pandas DataFrame.")

    if probs.shape[1] == 0:
        warnings.warn("No terminal states are detected", stacklevel=2)
        mudata.obs["shannon_entropy"] = pd.Series(np.nan, index=mudata.obs.index)
        mudata.obs["kl_divergence"] = pd.Series(np.nan, index=mudata.obs.index)
        return

    shannon_entropy = scipy.stats.entropy(probs.values, axis=1)
    average_distribution = probs.values.mean(axis=0)
    kl_divergence = np.nan_to_num(
        scipy.stats.entropy(probs.values, average_distribution, axis=1, base=2), nan=1.0, copy=False
    )
    shannon_entropy, kl_divergence = _minmax(shannon_entropy), _minmax(kl_divergence)
    mudata.obs["shannon_entropy"] = pd.Series(shannon_entropy, index=mudata.obs.index)
    mudata.obs["kl_divergence"] = pd.Series(kl_divergence, index=mudata.obs.index)


def _disambiguate_names(proposed: Mapping[str, str]) -> dict[str, str]:
    """Give every state a distinct name where several would take the same one.

    Follows CellRank, which appends ``_1``, ``_2`` … to a name claimed by more than one state
    and leaves a name claimed by one alone. Naming states after an annotation of the cells is
    then only a renaming: states are never combined because the annotation happens to give
    them the same name, which would let a naming choice change how many states there are.

    ``proposed`` is ordered, and the order decides which state takes which suffix, so
    repeating a run repeats the names.
    """
    claims: dict[str, int] = {}
    for name in proposed.values():
        claims[name] = claims.get(name, 0) + 1

    resolved, taken = {}, dict.fromkeys(claims, 0)
    for key, name in proposed.items():
        if claims[name] == 1:
            resolved[key] = name
            continue
        taken[name] += 1
        resolved[key] = f"{name}_{taken[name]}"

    return resolved


def migrate_states(mudata: MuData) -> None:
    """Record states written by an v1.0.0 in the form 1.1.0 reads.

    Before this layout, states were held only as dictionaries in ``.uns`` and colours as a
    single mapping. Neither is a form the wider ecosystem reads, and a stored key cannot warn
    when it is read, so an object saved then would otherwise acquire the current layout only
    by being computed again.

    The coarse states are reconstructed as the union of the initial, terminal and intermediate
    states recorded. That union is exactly what a fresh run writes: the coarse-graining where
    one was computed, since the intermediate states were every coarse-state cell marked
    neither initial nor terminal, and the union of the other kinds where none was.

    Nothing is removed. The dictionaries and the superseded colour mapping are left in place,
    so anything still reading them keeps working, and discarding them stays a separate choice.


    Parameters
    ----------
    mudata
        Multimodal annotated data object written by an earlier version.

    Returns
    -------
    None
        The object is annotated in place with the state columns, their colour lists and the
        record of the colour assignment.

    Raises
    ------
    KeyError
        If the object carries none of the superseded state dictionaries, there being nothing
        to convert.

    Examples
    --------
    >>> mudata = mudata.read_h5mu("saved_by_an_earlier_version.h5mu")
    >>> atlas.tl.migrate_states(mudata)
    """
    recorded = {
        kind: states
        for kind in ("initial_states", "terminal_states", "intermediate_states")
        if isinstance(states := mudata.uns.get(kind), Mapping) and states
    }

    if not recorded:
        raise KeyError(
            "none of `mudata.uns['initial_states']`, `['terminal_states']` or "
            "`['intermediate_states']` holds states, so there is nothing to convert; this "
            "object was not written by a version that recorded them there."
        )

    for kind in ("initial_states", "terminal_states"):
        mudata.obs[kind] = _states_to_column(recorded.get(kind, {}), mudata.obs_names)

    # Ordered so the coarse states read the way a fresh run writes them.
    union: dict[str, list[str]] = {}
    for kind in ("macrostates", "initial_states", "terminal_states", "intermediate_states"):
        for name, cells in recorded.get(kind, {}).items():
            union.setdefault(str(name), []).extend(cells)
    mudata.obs["macrostates"] = _states_to_column(union, mudata.obs_names)

    # Colours already chosen are carried over rather than assigned again, so migrating twice
    # cannot recolour anything. The superseded mapping has the shape of the record, so this
    # part is a copy.
    palette = mudata.uns.setdefault(_PALETTE_KEY, {})
    for name, color in (mudata.uns.get(_LEGACY_PALETTE_KEY) or {}).items():
        palette.setdefault(str(name), color)

    _assign_state_colors(mudata)
