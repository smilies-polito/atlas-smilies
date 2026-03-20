import numpy as np
import pytest
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import compute_kernel

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


def test_missing_distance_key():
    mdata = _create_mudata()
    with pytest.raises(KeyError, match="obsp"):
        compute_kernel(mdata, distance_key="wrong_key")


def test_missing_knn_key():
    mdata = _create_mudata()
    with pytest.raises(KeyError, match="uns"):
        compute_kernel(mdata, knn_key="wrong_key")


def test_knn_clipped_to_wnn():
    mdata1 = _create_mudata()
    mdata2 = _create_mudata()
    compute_kernel(mdata1, knn=None)
    compute_kernel(mdata2, knn=500)
    K1 = mdata1.obsp["DM_Kernel"]
    K2 = mdata2.obsp["DM_Kernel"]
    diff = (K1 - K2).data
    assert np.allclose(diff, 0)


def test_kernel_key():
    mdata = _create_mudata()
    compute_kernel(mdata, kernel_key="kernel")
    assert "kernel" in mdata.obsp
    assert mdata.obsp["kernel"].shape == (len(CELLS), len(CELLS))


def test_kernel_is_symmetric():
    mdata = _create_mudata()
    compute_kernel(mdata)
    kernel = mdata.obsp["DM_Kernel"]
    assert kernel.shape == (len(CELLS), len(CELLS))
    diff = (kernel - kernel.T).data
    assert np.allclose(diff, 0)


def test_kernel_non_negative():
    mdata = _create_mudata()
    compute_kernel(mdata)
    kernel = mdata.obsp["DM_Kernel"]
    assert kernel.shape == (len(CELLS), len(CELLS))
    assert np.all(kernel.data >= 0)


def test_alpha_changes_kernel():
    mdata1 = _create_mudata()
    mdata2 = _create_mudata()

    compute_kernel(mdata1, alpha=0)
    compute_kernel(mdata2, alpha=1)

    K1 = mdata1.obsp["DM_Kernel"]
    K2 = mdata2.obsp["DM_Kernel"]

    assert not np.allclose(K1.toarray(), K2.toarray())
