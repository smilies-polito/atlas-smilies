import numpy as np
import pytest
from anndata import AnnData
from muon import MuData

from atlas.pp import knn

SEED = 42
CELLS = [f"cell{i}" for i in range(60)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
N_NEIGHBORS, N_JOINT = 5, 10


def _create_mudata(with_rep: bool = True, rep_key: str = "X_joint") -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    if with_rep:
        mudata.obsm[rep_key] = rng.random((len(CELLS), N_JOINT))
    return mudata


def test_graph_is_built():
    result = knn(_create_mudata(), use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED)
    assert "joint_distances" in result.obsp
    assert "joint_connectivities" in result.obsp
    assert result.obsp["joint_distances"].shape == (len(CELLS), len(CELLS))


def test_recorded_neighbour_count():
    result = knn(_create_mudata(), use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED)
    assert result.uns["joint"]["params"]["n_neighbors"] == N_NEIGHBORS


def test_record_names_the_matrices():
    result = knn(_create_mudata(), use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED)
    record = result.uns["joint"]
    assert record["distances_key"] == "joint_distances"
    assert record["connectivities_key"] == "joint_connectivities"


def test_route_and_representation_are_recorded():
    result = knn(_create_mudata(), use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED)
    assert result.uns["joint"]["atlas"]["route"] == "representation"
    assert result.uns["joint"]["atlas"]["use_rep"] == "X_joint"


def test_modalities_are_untouched():
    mdata = _create_mudata()
    result = knn(mdata, use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED)
    assert set(result.mod) == {"rna", "activity"}
    for mod in ("rna", "activity"):
        assert "distances" not in result[mod].obsp


def test_key_added():
    mdata = _create_mudata()
    knn(mdata, use_rep="X_joint", n_neighbors=N_NEIGHBORS, key_added="graph", random_state=SEED)
    assert "graph_distances" in mdata.obsp
    assert mdata.uns["graph"]["distances_key"] == "graph_distances"


def test_copy_true():
    mdata = _create_mudata()
    result = knn(mdata, use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED, copy=True)
    assert result is not mdata
    assert "joint_distances" in result.obsp
    assert "joint_distances" not in mdata.obsp


def test_copy_false():
    mdata = _create_mudata()
    result = knn(mdata, use_rep="X_joint", n_neighbors=N_NEIGHBORS, random_state=SEED, copy=False)
    assert result is mdata
    assert "joint_distances" in mdata.obsp


def test_missing_representation():
    with pytest.raises(KeyError, match="X_missing"):
        knn(_create_mudata(with_rep=False), use_rep="X_missing")
