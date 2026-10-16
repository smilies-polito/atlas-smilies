import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cellrank.kernels import PrecomputedKernel, PseudotimeKernel
from muon import MuData
from scipy.sparse import csr_array, csr_matrix, issparse

from atlas.tl import CellRankExtension

SEED, K = 42, 10
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
CLUSTERS = ["red" if i <= 50 else "blue" for i in range(len(CELLS))]
PSEUDOTIME = [i / 100 for i in range(len(CELLS))]
SCHUR_COMPONENTS, N_CELLS = 4, 10
INITIAL_STATE = {CLUSTERS[0]: CELLS[:10]}
TERMINAL_STATE = {CLUSTERS[-1]: CELLS[-10:]}
CONN_KEY = "wnn_connectivities"
NORMALISED = "do not sum to 1"


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
    distances = 0.5 * (distances + distances.T)
    connectivities = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(len(CELLS), len(CELLS)))
    connectivities = 0.5 * (connectivities + connectivities.T)
    mudata.obsp["wnn_distances"] = distances
    mudata.obsp[CONN_KEY] = connectivities

    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}

    return mudata


def _pseudotime_kernel() -> PseudotimeKernel:
    """A pseudotime kernel computed in CellRank on the MuData's cells, in their order."""
    mudata = _create_mudata()
    adata = AnnData(obs=mudata.obs.copy())
    adata.obsp[CONN_KEY] = mudata.obsp[CONN_KEY]
    return PseudotimeKernel(adata, time_key="pseudotime", conn_key=CONN_KEY).compute_transition_matrix()


def _transition_matrix() -> csr_matrix:
    return csr_matrix(_pseudotime_kernel().transition_matrix)


def _run_fixed(cext: CellRankExtension) -> pd.DataFrame:
    cext.run(
        n_components=SCHUR_COMPONENTS,
        n_cells=N_CELLS,
        initial_states=INITIAL_STATE,
        terminal_states=TERMINAL_STATE,
        solver="direct",
        use_petsc=False,
    )
    return cext.mudata.obsm["fate_probabilities"]


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def _normalisation_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if NORMALISED in str(w.message)]


@pytest.mark.parametrize("convert", [_dense, csr_matrix, csr_array], ids=["ndarray", "csr_matrix", "csr_array"])
def test_a_matrix_is_stored_as_a_precomputed_kernel(convert):
    matrix = _transition_matrix()
    cext = CellRankExtension(_create_mudata())
    cext.set_kernel(convert(matrix))
    assert isinstance(cext.kernel, PrecomputedKernel)
    np.testing.assert_allclose(_dense(cext.kernel.transition_matrix), matrix.toarray())


def test_dense_and_sparse_give_the_same_result():
    matrix = _transition_matrix()
    dense, sparse = CellRankExtension(_create_mudata()), CellRankExtension(_create_mudata())
    dense.set_kernel(matrix.toarray())
    sparse.set_kernel(matrix)
    pd.testing.assert_frame_equal(_run_fixed(dense), _run_fixed(sparse))


@pytest.mark.parametrize("kernel", [pd.DataFrame(np.eye(len(CELLS))), np.eye(len(CELLS)).tolist()])
def test_an_unsupported_type_is_rejected(kernel):
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(TypeError, match="numpy.ndarray"):
        cext.set_kernel(kernel)


def test_a_cellrank_kernel_object_is_rejected():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(TypeError, match="found PseudotimeKernel"):
        cext.set_kernel(_pseudotime_kernel())
    assert not hasattr(cext, "_kernel")


def test_no_time_key_is_accepted():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(TypeError, match="time_key"):
        cext.set_kernel(_transition_matrix(), time_key="pseudotime")


def test_a_supplied_pseudotime_matrix_gives_the_result_of_computing_it():
    computed = CellRankExtension(_create_mudata())
    computed.compute_kernel(time_key="pseudotime", cluster_key="cluster")

    supplied = CellRankExtension(_create_mudata())
    supplied.set_kernel(_pseudotime_kernel().transition_matrix, cluster_key="cluster")

    pd.testing.assert_frame_equal(_run_fixed(computed), _run_fixed(supplied))
    for column in ("initial_states", "terminal_states", "macrostates"):
        pd.testing.assert_series_equal(computed.mudata.obs[column], supplied.mudata.obs[column])


def test_recorded_states_belong_to_the_mudata_cells():
    """A bare `PrecomputedKernel(matrix)` names its cells "0".."n-1", leaving these empty."""
    cext = CellRankExtension(_create_mudata())
    cext.set_kernel(_transition_matrix().toarray())
    _run_fixed(cext)
    for column in ("initial_states", "terminal_states"):
        recorded = cext.mudata.obs[column]
        assert recorded.index.equals(cext.mudata.obs_names)
        assert recorded.notna().any()


def test_a_matrix_of_the_wrong_shape_is_rejected():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(ValueError, match=r"\(100, 100\)"):
        cext.set_kernel(_transition_matrix()[:-1])


def test_a_matrix_direction_is_used_as_given():
    cext = CellRankExtension(_create_mudata())
    cext.set_kernel(_transition_matrix(), backward=True)
    assert cext.kernel.backward is True


def test_a_non_stochastic_matrix_warns_and_is_normalised():
    connectivities = _create_mudata().obsp[CONN_KEY]
    cext = CellRankExtension(_create_mudata())
    with pytest.warns(UserWarning, match=f"100 row\\(s\\) of the kernel {NORMALISED}"):
        cext.set_kernel(connectivities)
    np.testing.assert_allclose(np.asarray(cext.kernel.transition_matrix.sum(axis=1)).ravel(), 1.0)


def test_a_stochastic_matrix_does_not_warn():
    cext = CellRankExtension(_create_mudata())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cext.set_kernel(_transition_matrix())
    assert not _normalisation_warnings(caught)


def test_a_negative_entry_is_rejected():
    """CellRank alone accepts one when the rows already sum to 1."""
    matrix = np.full((len(CELLS), len(CELLS)), 1 / len(CELLS))
    matrix[0, 0], matrix[0, 1] = matrix[0, 0] + 0.5, matrix[0, 1] - 0.5
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(ValueError, match="negative"):
        cext.set_kernel(matrix)


def test_a_row_with_no_positive_entry_is_rejected():
    matrix = _transition_matrix().toarray()
    matrix[0] = 0
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(ValueError, match="no positive entry"):
        cext.set_kernel(matrix)


def test_a_missing_cluster_key_is_rejected():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(KeyError, match="fake_key"):
        cext.set_kernel(_transition_matrix(), cluster_key="fake_key")


def test_a_cluster_key_is_used_by_run():
    cext = CellRankExtension(_create_mudata())
    cext.set_kernel(_transition_matrix(), cluster_key="cluster")
    assert cext.cluster_key == "cluster"
    cext.run(n_components=SCHUR_COMPONENTS, n_cells=N_CELLS, solver="direct", use_petsc=False, allow_overlap=True)
    assert set(cext.mudata.uns["terminal_states"]).issubset(set(CLUSTERS))
    assert set(cext.mudata.uns["initial_states"]).issubset(set(CLUSTERS))


def test_setting_a_kernel_leaves_the_mudata_unchanged():
    mudata = _create_mudata()
    before = {name: list(getattr(mudata, name).keys()) for name in ("obs", "obsm", "obsp", "uns")}
    obs = mudata.obs.copy()
    CellRankExtension(mudata).set_kernel(_transition_matrix(), cluster_key="cluster")
    assert {name: list(getattr(mudata, name).keys()) for name in ("obs", "obsm", "obsp", "uns")} == before
    pd.testing.assert_frame_equal(mudata.obs, obs)


def test_a_failing_call_leaves_the_instance_as_it_was():
    cext = CellRankExtension(_create_mudata())
    cext.set_kernel(_transition_matrix(), cluster_key="cluster")
    kernel = cext.kernel
    with pytest.raises(ValueError):
        cext.set_kernel(_transition_matrix()[:-1])
    assert cext.kernel is kernel
    assert cext.cluster_key == "cluster"
