import numpy as np
import pytest
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import PalantirExtension

SEED, K = 42, 10
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]


def _create_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})

    # mock wnn distances
    rows = np.repeat(np.arange(len(CELLS)), K)
    cols = rng.integers(0, len(CELLS), size=len(CELLS) * K)
    dists = rng.random(len(CELLS) * K)

    distances = csr_matrix((dists, (rows, cols)), shape=(len(CELLS), len(CELLS)))
    connectivities = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.obsp["wnn_distances"] = distances
    mudata.obsp["wnn_connectivities"] = connectivities

    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}

    return mudata


def _complete_record_mudata() -> MuData:
    mdata = _create_mudata()
    mdata.uns["wnn"] = {
        "distances_key": "wnn_distances",
        "connectivities_key": "wnn_connectivities",
        "params": {"n_neighbors": K},
    }
    return mdata


def test_missing_distance_key():
    mdata = _create_mudata()
    pext = PalantirExtension(mudata=mdata)
    with pytest.warns(FutureWarning, match="distance_key"), pytest.raises(KeyError, match="obsp"):
        pext.compute_kernel(distance_key="wrong_key")


def test_missing_knn_key():
    mdata = _create_mudata()
    pext = PalantirExtension(mudata=mdata)
    with pytest.warns(FutureWarning, match="knn_key"), pytest.raises(KeyError, match="uns"):
        pext.compute_kernel(knn_key="wrong_key")


def test_knn_clipped_to_wnn():
    pext1 = PalantirExtension(_create_mudata())
    pext2 = PalantirExtension(_create_mudata())
    pext1.compute_kernel(knn=None)
    pext2.compute_kernel(knn=500)
    K1 = pext1.mudata.obsp["DM_Kernel"]
    K2 = pext2.mudata.obsp["DM_Kernel"]
    diff = (K1 - K2).data
    assert np.allclose(diff, 0)


def test_kernel_key():
    pext = PalantirExtension(_create_mudata())
    pext.compute_kernel(kernel_key="kernel")
    assert "kernel" in pext.mudata.obsp
    assert pext.mudata.obsp["kernel"].shape == (len(CELLS), len(CELLS))


def test_kernel_is_symmetric():
    pext = PalantirExtension(_create_mudata())
    pext.compute_kernel()
    kernel = pext.mudata.obsp["DM_Kernel"]
    assert kernel.shape == (len(CELLS), len(CELLS))
    diff = (kernel - kernel.T).data
    assert np.allclose(diff, 0)


def test_kernel_non_negative():
    pext = PalantirExtension(_create_mudata())
    pext.compute_kernel()
    kernel = pext.mudata.obsp["DM_Kernel"]
    assert kernel.shape == (len(CELLS), len(CELLS))
    assert np.all(kernel.data >= 0)


def _same_matrix(a, b) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


def test_key_alone_is_sufficient():
    mdata = _complete_record_mudata()
    PalantirExtension(mdata).compute_kernel(key="wnn")
    assert "DM_Kernel" in mdata.obsp


def test_default_key_is_unchanged():
    mdata = _complete_record_mudata()
    PalantirExtension(mdata).compute_kernel()
    assert "DM_Kernel" in mdata.obsp


def test_matrix_name_comes_from_the_record():
    mdata = _complete_record_mudata()
    mdata.obsp["oddly_named"] = mdata.obsp["wnn_distances"]
    mdata.uns["wnn"]["distances_key"] = "oddly_named"
    PalantirExtension(mdata).compute_kernel(key="wnn")
    assert "DM_Kernel" in mdata.obsp


def test_record_without_matrix_names_falls_back():
    mdata = _create_mudata()  # records only params
    PalantirExtension(mdata).compute_kernel(key="wnn")
    assert "DM_Kernel" in mdata.obsp


def test_unresolvable_key_is_named():
    mdata = _complete_record_mudata()
    with pytest.raises(KeyError, match="ghost"):
        PalantirExtension(mdata).compute_kernel(key="ghost")


@pytest.mark.parametrize(("old", "value"), [("knn_key", "wnn"), ("distance_key", "wnn_distances")])
def test_superseded_parameter_works_and_warns(old, value):
    mdata = _complete_record_mudata()
    with pytest.warns(FutureWarning, match=old):
        PalantirExtension(mdata).compute_kernel(**{old: value})
    assert "DM_Kernel" in mdata.obsp


@pytest.mark.parametrize("old", ["knn_key", "distance_key"])
def test_warning_names_replacement_and_removal_version(old):
    mdata = _complete_record_mudata()
    value = "wnn" if old == "knn_key" else "wnn_distances"
    with pytest.warns(FutureWarning) as record:
        PalantirExtension(mdata).compute_kernel(**{old: value})
    message = str(record[0].message)
    assert "`key`" in message
    assert "2.0.0" in message


def test_warning_is_attributed_to_the_caller():
    mdata = _complete_record_mudata()
    with pytest.warns(FutureWarning) as record:
        PalantirExtension(mdata).compute_kernel(knn_key="wnn")
    assert record[0].filename == __file__


@pytest.mark.parametrize(("old", "value"), [("knn_key", "wnn"), ("distance_key", "wnn_distances")])
def test_key_combined_with_a_superseded_parameter_raises(old, value):
    mdata = _complete_record_mudata()
    with pytest.raises(ValueError, match=old):
        PalantirExtension(mdata).compute_kernel(key="wnn", **{old: value})
