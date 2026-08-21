import matplotlib
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

import atlas
from atlas.tl import umap

matplotlib.use("Agg")

SEED, CELLS = 42, 40


def _representation_mudata(modalities: int = 1) -> MuData:
    rng = np.random.default_rng(SEED)
    mods = {}
    for i in range(modalities):
        mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
        mod.obs_names = [f"cell{j}" for j in range(CELLS)]
        mods[f"mod{i}"] = mod
    mudata = MuData(mods)
    mudata.obsm["X_joint"] = rng.random((CELLS, 8))
    atlas.pp.knn(mudata, use_rep="X_joint", n_neighbors=5, random_state=SEED)
    return mudata


def _wnn_mudata(extra_modality: bool = False) -> MuData:
    rng = np.random.default_rng(SEED)
    names = ["rna", "activity"] + (["spare"] if extra_modality else [])
    mods = {}
    for name in names:
        mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
        mod.obs_names = [f"cell{j}" for j in range(CELLS)]
        mods[name] = mod
    mudata = MuData(mods)
    for name in names:
        sc.pp.pca(mudata[name], n_comps=5, random_state=SEED)
    atlas.pp.wnn(mudata, knn=5, n_pcs=5, n_neighbors=5, n_multineighbors=5, n_bandwidth_neighbors=5, random_state=SEED)
    return mudata


def test_modality_outside_the_graph_is_ignored():
    mdata = _wnn_mudata(extra_modality=True)
    umap(mdata)
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)
    assert "spare" in mdata.mod


def test_modality_count_does_not_decide_the_route():
    mdata = _representation_mudata(modalities=3)
    umap(mdata)
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)


def test_result_is_stored_where_plotting_looks():
    for mdata in (_wnn_mudata(), _representation_mudata()):
        umap(mdata)
        assert "X_umap" in mdata.obsm


def test_single_graph_is_found():
    mdata = _representation_mudata()
    umap(mdata)
    assert "X_umap" in mdata.obsm


def test_graph_under_a_non_default_name_is_found():
    rng = np.random.default_rng(SEED)
    mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
    mod.obs_names = [f"cell{j}" for j in range(CELLS)]
    mdata = MuData({"mod0": mod})
    mdata.obsm["X_joint"] = rng.random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, key_added="whatever", random_state=SEED)

    umap(mdata)
    assert "X_umap" in mdata.obsm


def test_no_graph_names_the_entry_points_that_build_one():
    rng = np.random.default_rng(SEED)
    mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
    mdata = MuData({"mod0": mod})
    with pytest.raises(KeyError, match="atlas.pp.wnn"):
        umap(mdata)


def test_several_graphs_are_reported_rather_than_chosen_between():
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    with pytest.raises(ValueError, match="several") as excinfo:
        umap(mdata)
    assert "wnn" in str(excinfo.value)
    assert "joint" in str(excinfo.value)


def test_naming_a_graph_skips_detection():
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    umap(mdata, neighbors_key="joint")
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)


def test_unknown_name_is_reported():
    mdata = _representation_mudata()
    with pytest.raises(KeyError, match="ghost"):
        umap(mdata, neighbors_key="ghost")


def test_graph_without_provenance_is_refused_rather_than_inferred():
    mdata = _representation_mudata()
    del mdata.uns["joint"]["atlas"]

    with pytest.raises(KeyError, match="no record of how it was built"):
        umap(mdata, neighbors_key="joint")


@pytest.mark.parametrize(
    ("route", "use_rep"),
    [("wnn", "X_joint"), ("representation", {"rna": -1, "activity": -1})],
)
def test_contradictory_record_fails_here(route, use_rep):
    """Delegating one of these fails inside muon or scanpy with a message naming neither
    the graph nor the fix."""
    mdata = _representation_mudata()
    mdata.uns["joint"]["atlas"]["route"] = route
    mdata.uns["joint"]["params"]["use_rep"] = use_rep

    with pytest.raises(ValueError, match="joint"):
        umap(mdata, neighbors_key="joint")


def test_missing_representation_is_reported():
    mdata = _representation_mudata()
    del mdata.obsm["X_joint"]
    with pytest.raises(KeyError, match="X_joint"):
        umap(mdata, neighbors_key="joint")


def test_missing_modality_is_reported():
    mdata = _wnn_mudata()
    mdata.uns["wnn"]["params"]["use_rep"] = {"rna": -1, "ghost": -1}
    with pytest.raises(KeyError, match="ghost"):
        umap(mdata, neighbors_key="wnn")


def test_copy_leaves_the_input_untouched():
    mdata = _representation_mudata()
    returned = umap(mdata, copy=True)
    assert "X_umap" in returned.obsm
    assert "X_umap" not in mdata.obsm


def test_without_copy_the_input_is_annotated():
    mdata = _representation_mudata()
    returned = umap(mdata, copy=False)
    assert returned is mdata
    assert "X_umap" in mdata.obsm


def test_a_second_embedding_replaces_the_first():
    """The location is fixed, so this is documented rather than worked around."""
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    umap(mdata, neighbors_key="wnn")
    first = mdata.obsm["X_umap"].copy()
    umap(mdata, neighbors_key="joint")

    assert mdata.obsm["X_umap"].shape == first.shape
    assert not np.array_equal(mdata.obsm["X_umap"], first)
