import warnings
from collections.abc import Mapping

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


def _invert_assignment(assignment):
    if not isinstance(assignment.dtype, pd.CategoricalDtype):
        assignment = assignment.astype("category")

    inverted_assignment = {state: assignment.index[assignment == state].tolist() for state in assignment.cat.categories}
    return inverted_assignment


def _assign_state_colors(mudata: MuData, cmap: str = "tab20"):
    all_states = set()
    if "fate_state_colors" not in mudata.uns:
        mudata.uns["fate_state_colors"] = {}
    color_map = mudata.uns["fate_state_colors"]

    for key in ["terminal_states", "initial_states", "intermediate_states"]:
        states = mudata.uns.get(key, None)
        if isinstance(states, dict):
            all_states.update(states.keys())

    new_states = [s for s in all_states if s not in color_map]
    if not new_states:
        return

    base_colors = plt.get_cmap(cmap).colors
    for state in sorted(new_states):
        idx = len(color_map)
        color = base_colors[idx % len(base_colors)]
        color_map[state] = to_hex(color)


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
