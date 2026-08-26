from collections.abc import Mapping
from copy import deepcopy

import muon as mu
import numpy as np
import scanpy as sc
from anndata import AnnData
from muon import MuData

from .utils import _resolve_graph_key

_EMBEDDING_KEY = "X_umap"
_PROVENANCE_KEY = "atlas"
_WNN_ROUTE = "wnn"
_REPRESENTATION_ROUTE = "representation"
_PRODUCERS = "`atlas.pp.wnn` or `atlas.pp.knn`"


def _graph_keys(mudata: MuData) -> list[str]:
    found = []
    for key, record in mudata.uns.items():
        if not isinstance(record, Mapping):
            continue
        provenance = record.get(_PROVENANCE_KEY)
        if isinstance(provenance, Mapping) and provenance.get("route") is not None:
            found.append(key)
    return found


def _detect_graph_key(mudata: MuData) -> str:
    found = _graph_keys(mudata)
    if len(found) == 1:
        return found[0]

    if not found:
        raise KeyError(
            "no neighbor graph built by ATLAS was found in mudata.uns; "
            f"run {_PRODUCERS} first, or pass `neighbors_key` naming an existing graph"
        )

    raise ValueError(
        f"several neighbor graphs built by ATLAS were found ({', '.join(sorted(found))}); "
        "pass `neighbors_key` naming the one to embed"
    )


def _resolve_route(mudata: MuData, key: str) -> tuple[dict, str]:
    record = mudata.uns.get(key)
    if record is None:
        raise KeyError(f"'{key}' not in mudata.uns; no graph is stored under that name")

    provenance = record.get(_PROVENANCE_KEY) if isinstance(record, Mapping) else None
    route = provenance.get("route") if isinstance(provenance, Mapping) else None
    if route is None:
        raise KeyError(
            f"the graph '{key}' carries no record of how it was built, so it was not "
            f"produced by {_PRODUCERS}; only graphs from those entry points can be embedded here"
        )

    if route not in (_WNN_ROUTE, _REPRESENTATION_ROUTE):
        raise ValueError(f"the graph '{key}' records an unknown route '{route}'")

    # hand-assembled record can disagree, and delegating one fails deep
    # inside muon or scanpy with a message naming neither the graph nor the fix.
    use_rep = record.get("params", {}).get("use_rep")
    per_modality = isinstance(use_rep, Mapping)
    if route == _WNN_ROUTE and not per_modality:
        raise ValueError(
            f"the record for '{key}' says it was built over several modalities, but its "
            f"`use_rep` parameter is {type(use_rep).__name__} rather than one entry per modality"
        )
    if route == _REPRESENTATION_ROUTE and per_modality:
        raise ValueError(
            f"the record for '{key}' says it was built from a single representation, but its "
            "`use_rep` parameter names one per modality"
        )

    return dict(record), route


def _graph_matrices(mudata: MuData, key: str) -> tuple[str, str]:
    return tuple(_resolve_graph_key(mudata, key, kind) for kind in ("distances", "connectivities"))


def umap(
    mudata: MuData,
    neighbors_key: str | None = None,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: str | np.ndarray | None = "spectral",
    random_state: int = 42,
    a: float | None = None,
    b: float | None = None,
    method: str = "umap",
    copy: bool = False,
) -> MuData:
    """Embed a neighbor graph built by ATLAS with UMAP.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying a graph from :func:`atlas.pp.wnn` or
        :func:`atlas.pp.knn`.
    neighbors_key
        Key in ``mudata.uns`` naming the graph to embed. When ``None`` the graph is found:
        the object is searched for one built by those entry points.
    min_dist
        Minimum distance between embedded points, as in :func:`scanpy.tl.umap`.
    spread
        Scale of the embedded points, as in :func:`scanpy.tl.umap`.
    n_components
        Number of dimensions of the embedding.
    maxiter
        Number of optimization epochs, as in :func:`scanpy.tl.umap`.
    alpha
        Initial learning rate, as in :func:`scanpy.tl.umap`.
    gamma
        Weighting of negative samples, as in :func:`scanpy.tl.umap`.
    negative_sample_rate
        Negative samples per positive one, as in :func:`scanpy.tl.umap`.
    init_pos
        Initialization of the embedding, as in :func:`scanpy.tl.umap`.
    random_state
        Random seed used for reproducibility.
    a
        Parameter of the embedding's differentiable approximation, as in :func:`scanpy.tl.umap`.
    b
        Parameter of the embedding's differentiable approximation, as in :func:`scanpy.tl.umap`.
    method
        Implementation to use, as in :func:`scanpy.tl.umap`.
    copy
        Whether to annotate a copy and return it, leaving the input untouched.

    Returns
    -------
    The annotated object, or an annotated copy of it when ``copy`` is ``True``. The
    embedding is stored in ``.obsm["X_umap"]`` and its parameters in ``.uns["umap"]``.

    Raises
    ------
    KeyError
        If ``neighbors_key`` names no graph; if no graph built by ATLAS is found when it is
        ``None``; if the named graph carries no record of how it was built; or if the
        representation the record names is absent.
    ValueError
        If several graphs built by ATLAS are found and none was named, or if the record
        contradicts itself.

    Notes
    -----
    Only graphs from :func:`atlas.pp.wnn` and :func:`atlas.pp.knn` are supported.
    :func:`atlas.pp.preprocessing` computes its own embedding during its run and is not
    among them.

    The embedding is always stored under ``"X_umap"``.
    """
    data = mudata.copy() if copy else mudata

    key = _detect_graph_key(data) if neighbors_key is None else neighbors_key
    record, route = _resolve_route(data, key)
    distances_key, connectivities_key = _graph_matrices(data, key)

    shared = {
        "min_dist": min_dist,
        "spread": spread,
        "n_components": n_components,
        "maxiter": maxiter,
        "alpha": alpha,
        "gamma": gamma,
        "negative_sample_rate": negative_sample_rate,
        "init_pos": init_pos,
        "random_state": random_state,
        "a": a,
        "b": b,
        "method": method,
    }

    if route == _WNN_ROUTE:
        missing = [mod for mod in record["params"]["use_rep"] if mod not in data.mod]
        if missing:
            raise KeyError(f"the graph '{key}' was built over modalities {missing}, which are not in mudata.mod")

        mu.tl.umap(data, neighbors_key=key, copy=False, **shared)
        return data

    use_rep = record["params"]["use_rep"]
    if use_rep not in data.obsm:
        raise KeyError(f"the graph '{key}' was built from '{use_rep}', which is not in mudata.obsm")

    # scanpy operates on single-modality objects, so the embedding is computed on a
    # temporary one carrying the representation and the graph, then copied back.
    tmp = AnnData(obs=data.obs.copy())
    tmp.obsm[use_rep] = data.obsm[use_rep]
    tmp.obsp[distances_key] = data.obsp[distances_key]
    tmp.obsp[connectivities_key] = data.obsp[connectivities_key]
    tmp.uns[key] = deepcopy(record)
    tmp.uns[key]["distances_key"] = distances_key
    tmp.uns[key]["connectivities_key"] = connectivities_key

    sc.tl.umap(tmp, neighbors_key=key, copy=False, **shared)

    data.obsm[_EMBEDDING_KEY] = tmp.obsm[_EMBEDDING_KEY]
    data.uns["umap"] = tmp.uns["umap"]
    return data
