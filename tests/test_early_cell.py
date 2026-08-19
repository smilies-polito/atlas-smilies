import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData
from palantir.utils import CellNotFoundException
from scipy.sparse import csr_matrix

from atlas.tl import PalantirExtension

SEED, K, N_COMPS = 42, 10, 5
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]

# Deterministic multiscale space: component j is maximal at cell j and minimal at
# cell 99-j, so the only candidate cells are the first and last N_COMPS. The bulk is
# drawn from a bounded continuous range rather than held constant, so that the space
# stays non-degenerate and the fallback's own Palantir run is well posed.
MULTISCALE = np.random.default_rng(SEED).uniform(0.2, 0.8, (len(CELLS), N_COMPS))
for _j in range(N_COMPS):
    MULTISCALE[_j, _j] = 1.0
    MULTISCALE[len(CELLS) - 1 - _j, _j] = 0.0
MULTISCALE = pd.DataFrame(MULTISCALE, index=CELLS)

EDGE_CELLS = set(CELLS[:N_COMPS]) | set(CELLS[-N_COMPS:])
RARE_CELL = "cell50"

# "edge" sits at the extremes and is findable; "rare" is in the bulk and is not.
_labels = ["edge" if c in EDGE_CELLS else "bulk" for c in CELLS]
_labels[CELLS.index(RARE_CELL)] = "rare"
CLUSTERS = pd.DataFrame({"cluster": _labels}, index=CELLS)


def _create_mudata(multiscale_key: str = "multiscale") -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    mudata.obs = CLUSTERS.copy()

    rows = np.repeat(np.arange(len(CELLS)), K)
    cols = rng.integers(0, len(CELLS), size=len(CELLS) * K)
    dists = rng.random(len(CELLS) * K)
    mudata.obsp["wnn_distances"] = csr_matrix((dists, (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.obsp["wnn_connectivities"] = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}

    if multiscale_key is not None:
        mudata.obsm[multiscale_key] = MULTISCALE
    return mudata


def test_returns_cell_of_requested_type():
    pext = PalantirExtension(_create_mudata())
    cell = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale")
    assert cell in CELLS
    assert pext.mudata.obs.loc[cell, "cluster"] == "edge"


def test_returned_cell_is_accepted_by_run():
    pext = PalantirExtension(_create_mudata())
    cell = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale")
    pext.run(early_cell=cell, knn=10, num_waypoints=20, eigvec_multi_key="multiscale")
    assert "pseudotime" in pext.mudata.obs


def test_selection_is_reproducible():
    pext = PalantirExtension(_create_mudata())
    first = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale")
    second = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale")
    assert first == second


def test_multiscale_space_computed_when_absent():
    pext = PalantirExtension(_create_mudata(multiscale_key=None))
    cell = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="MULTISCALE")
    assert cell in CELLS
    assert "MULTISCALE" in pext.mudata.obsm
    assert "DM_Kernel" in pext.mudata.obsp


def test_unknown_cluster_key():
    pext = PalantirExtension(_create_mudata())
    with pytest.raises(KeyError, match="not_a_column"):
        pext.early_cell(celltype="edge", cluster_key="not_a_column", eigvec_multi_key="multiscale")


def test_absent_celltype():
    pext = PalantirExtension(_create_mudata())
    with pytest.raises(ValueError, match="ghost"):
        pext.early_cell(celltype="ghost", cluster_key="cluster", eigvec_multi_key="multiscale")


def test_not_found_without_fallback_raises():
    pext = PalantirExtension(_create_mudata())
    with pytest.raises(CellNotFoundException):
        pext.early_cell(celltype="rare", cluster_key="cluster", eigvec_multi_key="multiscale")


def test_fallback_warns_and_returns():
    pext = PalantirExtension(_create_mudata())
    with pytest.warns(UserWarning, match="costs as much as"):
        cell = pext.early_cell(
            celltype="rare", cluster_key="cluster", eigvec_multi_key="multiscale", fallback_seed=SEED
        )
    assert cell == RARE_CELL


def test_no_warning_when_fallback_is_not_reached(recwarn):
    # `fallback_seed` being set must not warn on its own: the warning belongs to the
    # fallback actually running, not to the possibility of it running. Guards the
    # attempt-cheap-path-first design against being simplified into an unconditional warn.
    pext = PalantirExtension(_create_mudata())
    cell = pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale", fallback_seed=SEED)
    assert cell in EDGE_CELLS
    assert [w for w in recwarn if "costs as much as" in str(w.message)] == []


def test_mudata_is_not_mutated():
    pext = PalantirExtension(_create_mudata())
    mdata = pext.mudata
    obs_before = mdata.obs.copy()
    obsm_before, obsp_before, uns_before = set(mdata.obsm), set(mdata.obsp), set(mdata.uns)

    pext.early_cell(celltype="edge", cluster_key="cluster", eigvec_multi_key="multiscale")

    assert mdata.obs.equals(obs_before)
    assert set(mdata.obsm) == obsm_before
    assert set(mdata.obsp) == obsp_before
    assert set(mdata.uns) == uns_before


def test_fallback_writes_nothing_back():
    pext = PalantirExtension(_create_mudata())
    mdata = pext.mudata
    obs_before = mdata.obs.copy()
    obsm_before = set(mdata.obsm)

    with pytest.warns(UserWarning):
        pext.early_cell(celltype="rare", cluster_key="cluster", eigvec_multi_key="multiscale", fallback_seed=SEED)

    assert mdata.obs.equals(obs_before)
    assert set(mdata.obsm) == obsm_before
