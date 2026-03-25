import muon as mu
import pandas as pd
import scanpy as sc
from muon import MuData


def preprocessing(
    mudata: MuData,
    n_pcs_rna: int = 30,
    n_pcs_act: int = 30,
    knn_rna: int = 30,
    knn_act: int = 30,
    use_rep: str | None = None,
    n_neighbors: int = 30,
    n_bandwidth_neighbors: int = 20,
    n_multineighbors: int = 200,
    metric: str = "euclidean",
    stranded: bool = False,
    fragment_path: str | None = None,
    features: pd.DataFrame | None = None,
    random_state: int = 42,
    copy: bool = False,
) -> MuData:
    """Preprocess multimodal single-cell data stored in a MuData object.

    This function runs a preprocessing workflow on a MuData object containing
    scRNA-seq data and either a precomputed gene activity modality or raw
    scATAC-seq data. If gene activity is not available, it is computed from
    ATAC-seq fragment data.

    The workflow includes:

    - Optional gene activity computation from scATAC-seq
    - Normalization and PCA on the activity modality
    - Construction of modality-specific k-nearest neighbor (kNN) graphs
    - Construction of a weighted nearest neighbor (WNN) graph
    - Computation of a multimodal UMAP embedding

    Parameters
    ----------
    mudata
            MuData object containing at least the ``"rna"`` modality and either
            an ``"activity"`` or ``"atac"`` modality.
    n_pcs_rna
            Number of principal components used for the RNA kNN graph.
    n_pcs_act
            Number of principal components used for the activity kNN graph.
    knn_rna
            Number of neighbors for the RNA kNN graph.
    knn_act
            Number of neighbors for the activity kNN graph.
    use_rep
            Key of the representation to use for neighbor graph construction,
            as in scanpy.pp.neighbors. If ``None``, the default representation is used.
    n_neighbors
            Number of neighbors for constructing the weighted nearest neighbor (WNN) graph,
            as in muon.pp.neighbors.
    n_bandwidth_neighbors
            Number of neighbors used for bandwidth estimation in the WNN graph,
            as in muon.pp.neighbors.
    n_multineighbors
            Number of neighbors used for multimodal neighbor construction,
            as in muon.pp.neighbors.
    metric
            Distance metric used for neighbor graph construction,
            as in muon.pp.neighbors.
    stranded
            Whether to consider strand information when computing gene activity.
    fragment_path
            Path to the fragment file used for gene activity computation if not
            already present in the ATAC modality. See muon.atac.tl.count_fragment_features
            for more infomation.
    random_state
            Random seed used for reproducibility.
    copy
            If ``True``, return a copy of the input MuData object. Otherwise,
            the input object is modified in place.


    Returns
    -------
    MuData
            MuData object restricted to the ``"rna"`` and ``"activity"`` modalities,
            containing:

            - modality-specific kNN graphs in ``.obsp``
            - weighted nearest neighbor graph stored under ``.uns["wnn"]``
            - multimodal UMAP embedding stored in ``.obsm["X_umap"]``

    Raises
    ------
    KeyError
            If required modalities are missing (e.g., ``"rna"`` or ``"atac"`` when
            gene activity must be computed).
    ValueError
            If required inputs for gene activity computation are not provided,
            such as fragment file or feature annotations.

    Examples
    --------
    >>> preprocess(mdata, n_pcs_rna=30, knn_rna=20)

    """
    data = MuData({k: v.copy() for k, v in mudata.mod.items()}) if copy else mudata

    if "rna" not in data.mod:
        raise KeyError("Modality 'rna' containing gene expression data is mandatory.")

    if "activity" not in data.mod:
        # Gene activity modality creation from scATAC-seq using muon.atac
        # 1. Verify "atac" modality exists
        # 2. Verify fragment file present in "uns"
        # 3. If not (2), then: (a) fragment file must be specified and (3) add fragment file to atac
        # 4. Mandatory features dataframe
        # 5. Compute activity
        # 6. Normalise activity
        # 7. Compute PCA
        if "atac" not in data.mod:
            raise KeyError("Modality 'atac' is mandatory when gene activity in not provided.")

        files = data.mod["atac"].uns.get("files", {})
        if not files or "fragments" not in files:
            if fragment_path is None:
                raise ValueError("Fragment file not found in `.uns['files']` and no `fragment_path` provided.")
            mu.atac.tl.locate_file(data.mod["atac"], file=fragment_path, key="fragments")
        if features is None:
            raise ValueError("Feature dataframe not provided.")
        data.mod["activity"] = mu.atac.tl.count_fragments_features(
            data=data.mod["atac"], features=features, stranded=stranded
        )
        # data.mod["activity"] = data.mod["activity"][data.mod["rna"].obs_names]
        sc.pp.normalize_total(data.mod["activity"])
        sc.pp.pca(data.mod["activity"], random_state=random_state)

    # Get MuData object only with gene expression and activity values if atac is present
    if "atac" in data.mod:
        del data.mod["atac"]
        data.update()

    # Compute KNN graphs
    if "distances" not in data.mod["rna"].obsp:
        sc.pp.neighbors(
            data.mod["rna"], n_neighbors=knn_rna, n_pcs=n_pcs_rna, random_state=random_state, use_rep=use_rep
        )
    if "distances" not in data.mod["activity"].obsp:
        sc.pp.neighbors(
            data.mod["activity"], n_neighbors=knn_act, n_pcs=n_pcs_act, random_state=random_state, use_rep=use_rep
        )

    mu.pp.neighbors(
        data,
        key_added="wnn",
        n_neighbors=n_neighbors,
        n_bandwidth_neighbors=n_bandwidth_neighbors,
        n_multineighbors=n_multineighbors,
        random_state=random_state,
        metric=metric,
    )
    mu.tl.umap(data, random_state=random_state, neighbors_key="wnn")
    return data
