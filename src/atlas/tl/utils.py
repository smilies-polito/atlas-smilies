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

_KEY_REMOVAL_VERSION = "2.0.0"
_STATE_KINDS = ("initial_states", "terminal_states", "macrostates")
_PALETTE_KEY = "atlas_state_palette"
_LEGACY_PALETTE_KEY = "fate_state_colors"


def _deprecated_key_arg(old_name: str, value, key, key_value) -> None:
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
    record = mudata.uns.get(key)
    name = record.get(f"{kind}_key") if isinstance(record, Mapping) else None
    if name is None:
        name = f"{key}_{kind}"

    if name not in mudata.obsp:
        raise KeyError(f"could not resolve the {kind} matrix for '{key}': '{name}' not in mudata.obsp")
    return name


def _cellrank_anndata(mudata: MuData, connectivity_key: str, cluster_key: str | None = None) -> AnnData:
    adata = AnnData(
        X=csr_matrix((mudata.n_obs, mudata["rna"].n_vars)),
        obs=mudata.obs.copy(),
        var=pd.DataFrame([], index=mudata["rna"].var_names),
    )
    adata.obsp[connectivity_key] = mudata.obsp[connectivity_key]

    if cluster_key is not None:
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    return adata


def _palantir_anndata(mudata: MuData, eigvec_key: str, multiscale: pd.DataFrame | np.ndarray) -> AnnData:
    adata = AnnData(obs=mudata.obs.copy())
    adata.obsm[eigvec_key] = multiscale
    return adata


def _states_to_column(states: Mapping[str, Sequence[str]], obs_names: pd.Index) -> pd.Series:
    assignment = pd.Series(pd.NA, index=obs_names, dtype=object)

    for name, cells in states.items():
        assignment.loc[list(cells)] = str(name)

    return assignment.astype(pd.CategoricalDtype(categories=[str(name) for name in states]))


def _resolve_overlap(initial: pd.Series, terminal: pd.Series) -> pd.Series:
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
    names: set[str] = set()

    for kind in _STATE_KINDS:
        column = mudata.obs.get(kind)
        if column is not None and isinstance(column.dtype, pd.CategoricalDtype):
            names.update(str(name) for name in column.cat.categories)

    for key in ("terminal_states", "initial_states", "intermediate_states"):
        states = mudata.uns.get(key)
        if isinstance(states, Mapping):
            names.update(str(name) for name in states)

    return names


def _guard_palette_collision(mudata: MuData) -> None:
    colliding = _LEGACY_PALETTE_KEY.removesuffix("_colors")
    if colliding in mudata.obs.columns:
        raise ValueError(
            f"`mudata.obs['{colliding}']` collides with `mudata.uns['{_LEGACY_PALETTE_KEY}']`, "
            f"which is a mapping: plotting that column would read the mapping positionally and "
            f"overwrite it, and `atlas.pl` still reads it. Rename the column."
        )


def _assign_state_colors(mudata: MuData, cmap: str = "tab20") -> None:
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

    union: dict[str, list[str]] = {}
    for kind in ("macrostates", "initial_states", "terminal_states", "intermediate_states"):
        for name, cells in recorded.get(kind, {}).items():
            union.setdefault(str(name), []).extend(cells)
    mudata.obs["macrostates"] = _states_to_column(union, mudata.obs_names)

    palette = mudata.uns.setdefault(_PALETTE_KEY, {})
    for name, color in (mudata.uns.get(_LEGACY_PALETTE_KEY) or {}).items():
        palette.setdefault(str(name), color)

    _assign_state_colors(mudata)


def _states_mapping(mudata: MuData, kind: str) -> dict[str, list[str]]:
    if kind in mudata.obs.columns:
        return _invert_assignment(mudata.obs[kind])

    superseded = mudata.uns.get(kind)
    if isinstance(superseded, Mapping) and superseded:
        warnings.warn(
            f"`mudata.uns['{kind}']` is superseded by `mudata.obs['{kind}']` and will be "
            f"removed in {_KEY_REMOVAL_VERSION}. It is being read for this call. Run "
            f"`atlas.tl.migrate_states` to record this object's states in the current form; "
            f"a stored key cannot announce this when it is read, so nothing else will.",
            FutureWarning,
            stacklevel=3,
        )
        return {str(name): list(cells) for name, cells in superseded.items()}

    return {}
