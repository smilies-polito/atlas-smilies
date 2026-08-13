import re

import numpy as np
import pandas as pd
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import PalantirExtension

SEED, K, NUM_WAYPOINTS = 42, 10, 20
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
EARLY_CELL = "cell0"
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
MULTISCALE = pd.DataFrame(np.random.default_rng(SEED).random((len(CELLS), 5)), index=CELLS)
EIGENVECTORS = pd.DataFrame(np.random.default_rng(SEED).random((len(CELLS), 5)), index=CELLS)
CLUSTERS = pd.DataFrame({"cluster": ["red" if i % 2 == 0 else "blue" for i in range(len(CELLS))]}, index=CELLS)


def _create_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    mudata.obs = CLUSTERS

    # mock wnn distances
    rows = np.repeat(np.arange(len(CELLS)), K)
    cols = rng.integers(0, len(CELLS), size=len(CELLS) * K)
    dists = rng.random(len(CELLS) * K)

    distances = csr_matrix((dists, (rows, cols)), shape=(len(CELLS), len(CELLS)))
    connectivities = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.obsp["wnn_distances"] = distances
    mudata.obsp["wnn_connectivities"] = connectivities

    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}

    # mock multiscale space
    mudata.obsm["multiscale"] = MULTISCALE
    mudata.obsm["eigenvectors"] = EIGENVECTORS
    return mudata


def test_non_available_multiscale():
    pext = PalantirExtension(_create_mudata())
    pext.run(early_cell=EARLY_CELL, knn=10, num_waypoints=NUM_WAYPOINTS, eigvec_multi_key="MULTISCALE")
    mdata = pext.mudata
    assert "DM_Kernel" in mdata.obsp
    assert "DM_Similarity" in mdata.obsp
    assert "DM_EigenVectors" in mdata.obsm
    assert mdata.obsm["DM_EigenVectors"].shape == (len(CELLS), 10)
    assert "DM_EigenValues" in mdata.uns
    assert len(mdata.uns["DM_EigenValues"]) == 10
    assert "MULTISCALE" in mdata.obsm


def test_run_with_barcodes():
    pext = PalantirExtension(_create_mudata())
    pext.run(
        early_cell=EARLY_CELL,
        knn=10,
        num_waypoints=NUM_WAYPOINTS,
        eigvec_key="eigenvectors",
        eigvec_multi_key="multiscale",
    )
    mdata = pext.mudata
    assert "pseudotime" in mdata.obs
    assert "kl_divergence" in mdata.obs
    assert "shannon_entropy" in mdata.obs
    assert "initial_states" in mdata.uns
    assert len(mdata.uns["initial_states"]) == 1
    assert "terminal_states" in mdata.uns
    # here cluster_key has no been specified, thereby uns["terminal_states"]
    # and uns["initial_states"] keys are obs_names
    for k, _ in mdata.uns["initial_states"].items():
        assert k in mdata.obs_names
    for k, _ in mdata.uns["terminal_states"].items():
        assert k in mdata.obs_names
    assert "fate_probabilities" in mdata.obsm


def test_run_with_clusters():
    pext = PalantirExtension(_create_mudata())
    pext.run(
        early_cell=EARLY_CELL,
        knn=10,
        cluster_key="cluster",
        num_waypoints=NUM_WAYPOINTS,
        eigvec_key="eigenvectors",
        eigvec_multi_key="multiscale",
    )
    mdata = pext.mudata
    assert "pseudotime" in mdata.obs
    assert "kl_divergence" in mdata.obs
    assert "shannon_entropy" in mdata.obs
    assert "initial_states" in mdata.uns
    assert len(mdata.uns["initial_states"]) == 1
    assert "terminal_states" in mdata.uns
    # `cluster_key` names states after the cluster and nothing else. Where several states
    # would take one cluster's name they are disambiguated with a `_1`, `_2` … suffix, so the
    # cluster is the stem of the name rather than the whole of it.
    for k, v in mdata.uns["initial_states"].items():
        assert re.sub(r"_\d+$", "", k) in ["red", "blue"]
        assert v[0] in mdata.obs_names
    for k, v in mdata.uns["terminal_states"].items():
        assert re.sub(r"_\d+$", "", k) in ["red", "blue"]
        assert v[0] in mdata.obs_names
    assert "fate_probabilities" in mdata.obsm


def test_states_are_recorded_as_columns_with_colours():
    """The layout the wider ecosystem reads, written alongside the superseded dictionaries."""
    pext = PalantirExtension(_create_mudata())
    pext.run(
        early_cell=EARLY_CELL,
        knn=10,
        cluster_key="cluster",
        num_waypoints=NUM_WAYPOINTS,
        eigvec_key="eigenvectors",
        eigvec_multi_key="multiscale",
    )
    mdata = pext.mudata

    for kind in ("initial_states", "terminal_states", "macrostates"):
        column = mdata.obs[kind]
        assert isinstance(column.dtype, pd.CategoricalDtype)
        assert column.notna().any()
        # one colour per category, in the categories' own order
        assert len(mdata.uns[f"{kind}_colors"]) == len(column.cat.categories)

    # `macrostates` is the union of the other kinds, Palantir computing no coarse-graining
    assert set(mdata.obs["macrostates"].cat.categories) == set(mdata.uns["initial_states"]) | set(
        mdata.uns["terminal_states"]
    )

    # a state named under both kinds carries one colour
    palette = mdata.uns["atlas_state_palette"]
    for kind in ("initial_states", "terminal_states", "macrostates"):
        for position, name in enumerate(mdata.obs[kind].cat.categories):
            assert mdata.uns[f"{kind}_colors"][position] == palette[name]
