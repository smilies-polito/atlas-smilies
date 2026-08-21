import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

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
    mudata.obsp["wnn_connectivities"] = connectivities

    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}

    return mudata


def test_missing_kernel():
    cext = CellRankExtension(_create_mudata())
    with pytest.raises(AttributeError, match="Kernel"):
        cext.run()


def test_without_cluster_key():
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime")
    cext.run(n_components=SCHUR_COMPONENTS, n_cells=N_CELLS, solver="direct", use_petsc=False, allow_overlap=True)
    assert hasattr(cext, "fate_key")
    assert cext.cluster_key is None
    mdata = cext.mudata
    assert "fate_probabilities" in mdata.obsm
    fate = mdata.obsm["fate_probabilities"]
    assert "terminal_states" in mdata.uns
    assert "initial_states" in mdata.uns
    assert "intermediate_states" in mdata.uns
    assert "kl_divergence" in mdata.obs
    assert "shannon_entropy" in mdata.obs
    terminal_keys = set(mdata.uns["terminal_states"].keys())
    initial_keys = set(mdata.uns["initial_states"].keys())
    clusters = set(CLUSTERS)
    assert not terminal_keys.issubset(clusters)
    assert not initial_keys.issubset(clusters)
    assert set(fate.columns) == terminal_keys


def test_with_cluster_key():
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime", cluster_key="cluster")
    cext.run(n_components=SCHUR_COMPONENTS, n_cells=N_CELLS, solver="direct", use_petsc=False, allow_overlap=True)
    assert hasattr(cext, "fate_key")
    mdata = cext.mudata
    assert "fate_probabilities" in mdata.obsm
    fate = mdata.obsm["fate_probabilities"]
    assert "terminal_states" in mdata.uns
    assert "initial_states" in mdata.uns
    assert "intermediate_states" in mdata.uns
    assert "kl_divergence" in mdata.obs
    assert "shannon_entropy" in mdata.obs
    terminal_keys = set(mdata.uns["terminal_states"].keys())
    initial_keys = set(mdata.uns["initial_states"].keys())
    clusters = set(CLUSTERS)
    assert terminal_keys.issubset(clusters)
    assert initial_keys.issubset(clusters)
    assert set(fate.columns) == terminal_keys


def test_fixed_states():
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime", cluster_key="cluster")
    cext.run(
        n_components=SCHUR_COMPONENTS,
        n_cells=N_CELLS,
        initial_states=INITIAL_STATE,
        terminal_states=TERMINAL_STATE,
        solver="direct",
        use_petsc=False,
    )
    mdata = cext.mudata
    assert len(mdata.uns["terminal_states"]) == 1
    assert len(mdata.uns["initial_states"]) == 1
    assert mdata.uns["intermediate_states"] == {}
    terminal_keys = set(mdata.uns["terminal_states"].keys())
    initial_keys = set(mdata.uns["initial_states"].keys())
    assert terminal_keys == set(TERMINAL_STATE.keys())
    assert initial_keys == set(INITIAL_STATE.keys())
    fate = mdata.obsm["fate_probabilities"]
    assert set(fate.columns) == set(TERMINAL_STATE.keys())


def test_states_are_recorded_as_columns_with_colours():
    """The layout the wider ecosystem reads, written alongside the superseded dictionaries."""
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime", cluster_key="cluster")
    cext.run(n_components=SCHUR_COMPONENTS, n_cells=N_CELLS, solver="direct", use_petsc=False, allow_overlap=True)
    mdata = cext.mudata

    for kind in ("initial_states", "terminal_states", "macrostates"):
        column = mdata.obs[kind]
        assert isinstance(column.dtype, pd.CategoricalDtype)
        assert column.notna().any()
        assert len(mdata.uns[f"{kind}_colors"]) == len(column.cat.categories)

    palette = mdata.uns["atlas_state_palette"]
    for kind in ("initial_states", "terminal_states", "macrostates"):
        for position, name in enumerate(mdata.obs[kind].cat.categories):
            assert mdata.uns[f"{kind}_colors"][position] == palette[name]

    # every fate-probability column names a terminal state that carries a colour, which is
    # what the plotting entry points index by
    for column in mdata.obsm["fate_probabilities"].columns:
        assert column in mdata.uns["fate_state_colors"]


def test_the_coarse_states_are_recorded_on_the_fixed_states_branch():
    """`compute_macrostates` is never called there, so the union of the supplied states is
    recorded instead, keeping the column present however the object was produced."""
    cext = CellRankExtension(_create_mudata())
    cext.compute_kernel(time_key="pseudotime", cluster_key="cluster")
    cext.run(
        n_components=SCHUR_COMPONENTS,
        n_cells=N_CELLS,
        solver="direct",
        use_petsc=False,
        initial_states=INITIAL_STATE,
        terminal_states=TERMINAL_STATE,
    )
    mdata = cext.mudata

    assert "macrostates" in mdata.obs
    assert set(mdata.obs["macrostates"].cat.categories) == set(mdata.uns["initial_states"]) | set(
        mdata.uns["terminal_states"]
    )
    assert mdata.uns["intermediate_states"] == {}
