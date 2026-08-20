import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cellrank.kernels import PseudotimeKernel
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import CellRankExtension

SEED, K = 42, 10
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
CLUSTERS = ["red" if i % 2 == 0 else "blue" for i in range(len(CELLS))]
PSEUDOTIME = [i / 100 for i in range(len(CELLS))]


def _create_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    mudata.obs = pd.DataFrame({"cluster": CLUSTERS, "pseudotime": PSEUDOTIME}, index=CELLS)

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


def test_missing_connectivity_key():
    cext = CellRankExtension(_create_mudata())
    # still works as it did, now announcing that `connectivity_key` is superseded
    with pytest.warns(FutureWarning, match="connectivity_key"), pytest.raises(KeyError, match="obsp"):
        cext.compute_kernel(connectivity_key="fake_key")


def test_missing_pseudotime_key():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(KeyError, match="obs"):
        cext.compute_kernel(time_key="fake_key")


def test_wrong_cluster_key():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(KeyError, match="obs"):
        cext.compute_kernel(time_key="pseudotime", cluster_key="fake_key")


def test_correct_behavior():
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime", cluster_key="cluster")
    assert hasattr(cext, "kernel")
    assert hasattr(cext, "_time_key")
    assert hasattr(cext, "_cluster_key")
    assert cext.time_key == "pseudotime"
    assert cext.cluster_key == "cluster"
    kernel = cext.kernel
    assert isinstance(kernel, PseudotimeKernel)
    assert hasattr(kernel, "transition_matrix")
    assert kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_key_alone_is_sufficient():
    cext = CellRankExtension(_complete_record_mudata())
    cext.compute_kernel(key="wnn", time_key="pseudotime", n_jobs=1)
    assert cext.kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_default_key_is_unchanged():
    cext = CellRankExtension(_complete_record_mudata())
    cext.compute_kernel(time_key="pseudotime", n_jobs=1)
    assert cext.kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_matrix_name_comes_from_the_record():
    mdata = _complete_record_mudata()
    mdata.obsp["oddly_named"] = mdata.obsp["wnn_connectivities"]
    mdata.uns["wnn"]["connectivities_key"] = "oddly_named"
    cext = CellRankExtension(mdata)
    cext.compute_kernel(key="wnn", time_key="pseudotime", n_jobs=1)
    assert cext.kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_record_without_matrix_names_falls_back():
    cext = CellRankExtension(_create_mudata())  # records only params
    cext.compute_kernel(key="wnn", time_key="pseudotime", n_jobs=1)
    assert cext.kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_unresolvable_key_is_named():
    with pytest.raises(KeyError, match="ghost"):
        CellRankExtension(_complete_record_mudata()).compute_kernel(key="ghost", time_key="pseudotime")


def test_superseded_parameter_works_and_warns():
    cext = CellRankExtension(_complete_record_mudata())
    with pytest.warns(FutureWarning, match="connectivity_key"):
        cext.compute_kernel(connectivity_key="wnn_connectivities", time_key="pseudotime", n_jobs=1)
    assert cext.kernel.transition_matrix.shape == (len(CELLS), len(CELLS))


def test_warning_names_replacement_and_removal_version():
    cext = CellRankExtension(_complete_record_mudata())
    with pytest.warns(FutureWarning) as record:
        cext.compute_kernel(connectivity_key="wnn_connectivities", time_key="pseudotime", n_jobs=1)
    message = str(record[0].message)
    assert "`key`" in message
    assert "2.0.0" in message


def test_key_combined_with_the_superseded_parameter_raises():
    with pytest.raises(ValueError, match="connectivity_key"):
        CellRankExtension(_complete_record_mudata()).compute_kernel(
            key="wnn", connectivity_key="wnn_connectivities", time_key="pseudotime"
        )
