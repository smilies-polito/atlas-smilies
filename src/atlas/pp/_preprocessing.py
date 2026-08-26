from collections.abc import Mapping, Sequence

import muon as mu
import pandas as pd
import scanpy as sc
from anndata import AnnData
from muon import MuData

from .utils import _safe_mudata


def _per_modality(value, modalities: Sequence[str], name: str) -> dict:
    if isinstance(value, Mapping):
        missing = [m for m in modalities if m not in value]
        if missing:
            raise KeyError(f"{name} does not provide a value for {missing}")
        return {m: value[m] for m in modalities}
    return dict.fromkeys(modalities, value)


def _write_graph(target: MuData, source, key: str, route: str, **provenance) -> None:
    """Store a neighbour graph on ``target``, taking it from ``source``.

    This is the only place the graph contract is written. Every route ends here, so that
    the keys, the record and the provenance cannot drift between them.

    ``source`` is the object the graph was computed on: a restricted ``MuData`` for the
    weighted route, a temporary ``AnnData`` for the representation route. No embedding is
    written; neighbour construction and embedding are separate steps.

    Raises
    ------
    KeyError
        If ``source`` carries no record under ``key``.
    ValueError
        If ``source`` and ``target`` do not agree on their observations, which would leave
        the graph's indices silently misaligned with the object.

    """
    if key not in source.uns:
        raise KeyError(f"{key} not in the computed object's .uns")

    if list(source.obs_names) != list(target.obs_names):
        raise ValueError(
            f"the object the graph was computed on carries {len(source.obs_names)} observations "
            f"in a different order or number from the {len(target.obs_names)} of the object being "
            "annotated; storing it would misalign the graph"
        )

    record = dict(source.uns[key])
    for kind in ("distances_key", "connectivities_key"):
        name = record[kind]
        target.obsp[name] = source.obsp[name]

    record["atlas"] = {"route": route, **provenance}
    target.uns[key] = record


def compute_gene_activity(
    mudata: MuData,
    features: pd.DataFrame | None = None,
    fragment_path: str | None = None,
    stranded: bool = False,
    count_reads: bool = False,
    atac_key: str = "atac",
    out_key: str = "activity",
    random_state: int = 42,
    copy: bool = False,
    n_comps: int | None = None,
) -> MuData:
    """Derive a gene activity modality from chromatin accessibility.

    This function exploits :func:`muon.atac.tl.count_fragments_features`.
    Results are normalized and PCA is computed.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying the accessibility modality.
    features
        Feature annotation. See :func:`muon.atac.tl.count_fragments_features` for more information.
    fragment_path
        Path to the fragment file for gene activity computation. See :func:`muon.atac.tl.count_fragments_features for more information.
        modality.
    stranded
        Whether to consider strand information when computing gene activity.
    count_reads
        Parameter for :func:`muon.atac.tl.count_fragments_features` determining which columns in the fragment file to use for feature aggregation.
    atac_key
        Modality holding chromatin accessibility.
    out_key
        Modality the derived activity is stored under.
    random_state
        Random seed used for the reduction.
    copy
        Whether to annotate a copy and return it, leaving the input untouched.
    n_comps
        Number of principal components to compute. Default ``None`` relies on standard
        computations in :func:`scanpy.pp.pca`.

    Returns
    -------
    The annotated object, or an annotated copy of it when ``copy`` is ``True``.

    Raises
    ------
    KeyError
        If ``atac_key`` is not a modality of ``mudata``.
    ValueError
        If the fragment file cannot be located, or ``features`` is not provided.
    ImportError
        If :mod:`pysam` is not installed. The error is raised by :mod:`muon` and carries
        its own installation instructions; see Notes.

    Notes
    -----
    Reading the fragment file requires :mod:`pysam`, which neither ATLAS nor :mod:`muon`
    installs by default.

    """
    data = mudata.copy() if copy else mudata

    if atac_key not in data.mod:
        raise KeyError(f"Modality '{atac_key}' containing chromatin accessibility is mandatory.")

    files = data.mod[atac_key].uns.get("files", {})
    if not files or "fragments" not in files:
        if fragment_path is None:
            raise ValueError("Fragment file not found in `.uns['files']` and no `fragment_path` provided.")
        mu.atac.tl.locate_file(data.mod[atac_key], file=fragment_path, key="fragments")

    if features is None:
        raise ValueError("Feature dataframe not provided.")

    data.mod[out_key] = mu.atac.tl.count_fragments_features(
        data=data.mod[atac_key], features=features, stranded=stranded, count_reads=count_reads
    )
    sc.pp.normalize_total(data.mod[out_key])
    sc.pp.pca(data.mod[out_key], n_comps=n_comps, random_state=random_state)
    return data


def wnn(
    mudata: MuData,
    modalities: Sequence[str] = ("rna", "activity"),
    knn: int | Mapping[str, int] = 15,
    n_pcs: int | Mapping[str, int] = 50,
    use_rep: str | None | Mapping[str, str | None] = None,
    n_neighbors: int | None = 30,
    n_bandwidth_neighbors: int = 20,
    n_multineighbors: int = 200,
    metric: str = "euclidean",
    key_added: str = "wnn",
    random_state: int = 42,
    copy: bool = False,
) -> MuData:
    """Build a weighted nearest neighbor graph over two modalities.

    The modalities named take part in the graph; any others the object carries are left
    alone and are neither used nor removed. Restriction happens on a throwaway object,
    since :func:`muon.pp.neighbors` uses every modality of the object it is given.

    Parameters
    ----------
    mudata
        Multimodal annotated data object.
    modalities
        Modalities to integrate.
    knn
        Neighborhood size for the modality graphs. A single value applies to every
        modality; a mapping supplies each one its own.
    n_pcs
        Number of components used for the modality graphs, as ``knn``.
    use_rep
        Representation used for the modality graphs, as ``knn``.
    n_neighbors
        Number of neighbors for the weighted graph, as in :func:`muon.pp.neighbors`. If
        ``None``, the arithmetic mean of the modality neighborhood sizes is used.
    n_bandwidth_neighbors
        Neighbors used for bandwidth estimation, as in :func:`muon.pp.neighbors`.
    n_multineighbors
        Neighbors used for multimodal neighbor construction, as in :func:`muon.pp.neighbors`.
    metric
        Distance metric, as in :func:`muon.pp.neighbors`.
    key_added
        Key the graph is stored under. See :func:`muon.pp.neighbors` for more information.
    random_state
        Random seed used for reproducibility.
    copy
        Whether to annotate a copy and return it, leaving the input untouched.

    Returns
    -------
    The annotated object, or an annotated copy of it when ``copy`` is ``True``.

    Raises
    ------
    KeyError
        If a named modality is not present, or a mapping omits one of them.

    Notes
    -----
    A modality that already carries a neighbor graph keeps it rather than having one
    recomputed.

    """
    data = mudata.copy() if copy else mudata

    missing = [m for m in modalities if m not in data.mod]
    if missing:
        raise KeyError(f"Modalities {missing} not in mudata.mod")

    knn_by_mod = _per_modality(knn, modalities, "knn")
    n_pcs_by_mod = _per_modality(n_pcs, modalities, "n_pcs")
    use_rep_by_mod = _per_modality(use_rep, modalities, "use_rep")

    for mod in modalities:
        if "distances" not in data.mod[mod].obsp:
            sc.pp.neighbors(
                data.mod[mod],
                n_neighbors=knn_by_mod[mod],
                n_pcs=n_pcs_by_mod[mod],
                random_state=random_state,
                use_rep=use_rep_by_mod[mod],
            )

    # muon integrates every modality of the object it is given, so restriction is done
    # here rather than on the caller's object.
    subset = _safe_mudata(data, modalities=list(modalities))
    mu.pp.neighbors(
        subset,
        key_added=key_added,
        n_neighbors=n_neighbors,
        n_bandwidth_neighbors=n_bandwidth_neighbors,
        n_multineighbors=n_multineighbors,
        random_state=random_state,
        metric=metric,
    )

    _write_graph(data, subset, key_added, route="wnn", modalities=list(modalities))
    return data


def knn(
    mudata: MuData,
    use_rep: str,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    key_added: str = "joint",
    random_state: int = 42,
    copy: bool = False,
) -> MuData:
    """Build a nearest neighbor graph over an already integrated representation via :func:`scanpy.pp.neighbors`

    Intended for a precombined representatio stored in ``mudata.obsm``.

    There is a single view, so no weighting is performed and
    no per-modality weights are produced.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying the representation in ``.obsm``.
    use_rep
        Key in ``mudata.obsm`` holding the representation.
    n_neighbors
        Neighborhood size.
    metric
        Distance metric, as in :func:`scanpy.pp.neighbors`.
    key_added
        Key the graph is stored under. See :func:`scanpy.pp.neighbors` for more information.
    random_state
        Random seed used for reproducibility.
    copy
        Whether to annotate a copy and return it, leaving the input untouched.

    Returns
    -------
    The annotated object, or an annotated copy of it when ``copy`` is ``True``.

    Raises
    ------
    KeyError
        If ``use_rep`` is not present in ``mudata.obsm``.

    """
    data = mudata.copy() if copy else mudata

    if use_rep not in data.obsm:
        raise KeyError(f"{use_rep} not in mudata.obsm")

    # scanpy operates on single-modality objects, so the graph is computed on a temporary
    # one carrying only the representation and then stored on the caller's object.
    tmp = AnnData(obs=data.obs.copy())
    tmp.obsm[use_rep] = data.obsm[use_rep]
    sc.pp.neighbors(
        tmp,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        metric=metric,
        random_state=random_state,
        key_added=key_added,
    )

    _write_graph(data, tmp, key_added, route="representation", use_rep=use_rep)
    return data
