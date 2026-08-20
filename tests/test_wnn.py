import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

from atlas.pp import wnn

SEED = 42
CELLS = [f"cell{i}" for i in range(60)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
PEAKS = [f"peak{i}" for i in range(30)]

KNN, N_PCS, N_NEIGHBORS, N_MULTI, N_BANDWIDTH, N_COMPS = 5, 5, 5, 30, 5, 10


def _create_mudata(with_atac: bool = True, extra_mod: bool = False) -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR
    mods = {"rna": rna, "activity": act}

    if with_atac:
        atac = AnnData(rng.random((len(CELLS), len(PEAKS))))
        atac.obs_names, atac.var_names = CELLS, PEAKS
        mods["atac"] = atac
    if extra_mod:
        other = AnnData(rng.random((len(CELLS), 8)))
        other.obs_names, other.var_names = CELLS, [f"o{i}" for i in range(8)]
        mods["other"] = other
    return MuData(mods)


def _prepared(**kwargs) -> MuData:
    mdata = _create_mudata(**kwargs)
    for mod in ("rna", "activity"):
        sc.pp.normalize_total(mdata[mod])
        sc.pp.pca(mdata[mod], n_comps=N_COMPS, random_state=SEED)
    return mdata


def test_graph_is_built():
    result = wnn(
        _prepared(),
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert "wnn_distances" in result.obsp
    assert "wnn_connectivities" in result.obsp
    assert result.uns["wnn"]["params"]["n_neighbors"] == N_NEIGHBORS


def test_any_modality_pair():
    mdata = _create_mudata()
    sc.pp.normalize_total(mdata["rna"])
    sc.pp.pca(mdata["rna"], n_comps=N_COMPS, random_state=SEED)
    sc.pp.pca(mdata["atac"], n_comps=N_COMPS, random_state=SEED)
    result = wnn(
        mdata,
        modalities=("rna", "atac"),
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert "wnn_distances" in result.obsp
    assert result.uns["wnn"]["atlas"]["modalities"] == ["rna", "atac"]


def test_scalar_settings_apply_to_every_modality():
    mdata = _prepared()
    wnn(
        mdata,
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    for mod in ("rna", "activity"):
        assert mdata[mod].uns["neighbors"]["params"]["n_neighbors"] == KNN


def test_per_modality_settings():
    mdata = _prepared()
    wnn(
        mdata,
        knn={"rna": 4, "activity": 7},
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert mdata["rna"].uns["neighbors"]["params"]["n_neighbors"] == 4
    assert mdata["activity"].uns["neighbors"]["params"]["n_neighbors"] == 7


def test_existing_modality_graph_is_reused():
    mdata = _prepared()
    sc.pp.neighbors(mdata["rna"], n_neighbors=9, n_pcs=N_PCS, random_state=SEED)
    wnn(mdata, knn=KNN, n_pcs=N_PCS, n_multineighbors=N_MULTI, n_bandwidth_neighbors=N_BANDWIDTH, random_state=SEED)
    assert mdata["rna"].uns["neighbors"]["params"]["n_neighbors"] == 9


def test_modalities_are_not_removed():
    mdata = _prepared(extra_mod=True)
    result = wnn(
        mdata,
        modalities=("rna", "activity"),
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert set(result.mod) == {"rna", "activity", "atac", "other"}


def test_route_is_recorded():
    result = wnn(
        _prepared(),
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert result.uns["wnn"]["atlas"]["route"] == "wnn"


def test_no_embedding_is_computed():
    result = wnn(
        _prepared(),
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert "X_umap" not in result.obsm


def test_key_added():
    mdata = _prepared()
    wnn(
        mdata,
        knn=KNN,
        n_pcs=N_PCS,
        key_added="graph",
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
    )
    assert "graph_distances" in mdata.obsp
    assert mdata.uns["graph"]["distances_key"] == "graph_distances"


def test_copy_true():
    mdata = _prepared()
    result = wnn(
        mdata,
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
        copy=True,
    )
    assert result is not mdata
    assert "wnn_distances" in result.obsp
    assert "wnn_distances" not in mdata.obsp


def test_copy_false():
    mdata = _prepared()
    result = wnn(
        mdata,
        knn=KNN,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        n_multineighbors=N_MULTI,
        n_bandwidth_neighbors=N_BANDWIDTH,
        random_state=SEED,
        copy=False,
    )
    assert result is mdata
    assert "wnn_distances" in mdata.obsp


def test_missing_modality():
    with pytest.raises(KeyError, match="ghost"):
        wnn(_prepared(), modalities=("rna", "ghost"))


def test_mapping_missing_a_modality():
    with pytest.raises(KeyError, match="knn"):
        wnn(_prepared(), modalities=("rna", "activity"), knn={"rna": KNN})
