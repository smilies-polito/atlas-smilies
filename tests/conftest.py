import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

from atlas.pp import knn, preprocessing, wnn


@pytest.fixture
def adata():
    adata = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)

    return adata


# --------------------------------------------------------------------------------------
# graphs built by each of the routes that can produce one
#
# The trajectory inference entry points must consume any of these identically, so both
# `compute_kernel` test modules need all four. They are built once per session because
# one of them runs a full preprocessing pass, which dominates the suite otherwise.
# --------------------------------------------------------------------------------------

_SEED = 42
_CELLS = [f"cell{i}" for i in range(60)]
_GENES = [f"gene{i}" for i in range(20)]
_ACT_VAR = [f"act{i}" for i in range(15)]
_KNN, _N_PCS, _N_NEIGHBORS, _N_COMPS, _N_MULTI, _N_BANDWIDTH = 5, 5, 5, 10, 5, 5


def _base(with_activity: bool = True) -> MuData:
    rng = np.random.default_rng(_SEED)
    rna = AnnData(rng.random((len(_CELLS), len(_GENES))))
    rna.obs_names, rna.var_names = _CELLS, _GENES
    mods = {"rna": rna}
    if with_activity:
        act = AnnData(rng.random((len(_CELLS), len(_ACT_VAR))))
        act.obs_names, act.var_names = _CELLS, _ACT_VAR
        mods["activity"] = act
    mudata = MuData(mods)
    for mod in mods:
        sc.pp.normalize_total(mudata[mod])
        sc.pp.pca(mudata[mod], n_comps=_N_COMPS, random_state=_SEED)
    mudata.obs["pseudotime"] = np.linspace(0, 1, len(_CELLS))
    return mudata


def _via_preprocessing() -> tuple[MuData, str]:
    mudata = preprocessing(
        _base(),
        knn_rna=_KNN,
        knn_act=_KNN,
        n_pcs_rna=_N_PCS,
        n_pcs_act=_N_PCS,
        n_neighbors=_N_NEIGHBORS,
        n_multineighbors=_N_MULTI,
        n_bandwidth_neighbors=_N_BANDWIDTH,
        random_state=_SEED,
    )
    mudata.obs["pseudotime"] = np.linspace(0, 1, len(_CELLS))
    return mudata, "wnn"


def _via_wnn() -> tuple[MuData, str]:
    mudata = _base()
    wnn(
        mudata,
        knn=_KNN,
        n_pcs=_N_PCS,
        n_neighbors=_N_NEIGHBORS,
        n_multineighbors=_N_MULTI,
        n_bandwidth_neighbors=_N_BANDWIDTH,
        random_state=_SEED,
    )
    return mudata, "wnn"


def _via_knn() -> tuple[MuData, str]:
    mudata = _base(with_activity=False)
    mudata.obsm["X_joint"] = np.random.default_rng(_SEED).random((len(_CELLS), 10))
    knn(mudata, use_rep="X_joint", n_neighbors=_N_NEIGHBORS, random_state=_SEED)
    return mudata, "joint"


def _via_custom_key() -> tuple[MuData, str]:
    mudata = _base()
    wnn(
        mudata,
        knn=_KNN,
        n_pcs=_N_PCS,
        n_neighbors=_N_NEIGHBORS,
        n_multineighbors=_N_MULTI,
        n_bandwidth_neighbors=_N_BANDWIDTH,
        key_added="graph",
        random_state=_SEED,
    )
    return mudata, "graph"


ROUTE_NAMES = ["preprocessing", "wnn", "knn", "custom_key"]


@pytest.fixture(scope="session")
def _graph_routes() -> dict[str, tuple[MuData, str]]:
    """One graph per producing route, as ``(mudata, key)``.

    Session-scoped: the entry points under test only read the graph and write their own
    results, so sharing is safe, and rebuilding per test would dominate the suite.
    """
    return {
        "preprocessing": _via_preprocessing(),
        "wnn": _via_wnn(),
        "knn": _via_knn(),
        "custom_key": _via_custom_key(),
    }


@pytest.fixture(params=ROUTE_NAMES, scope="session")
def graph_route(request, _graph_routes) -> tuple[MuData, str]:
    """A graph from one producing route, parametrised over all of them."""
    return _graph_routes[request.param]
