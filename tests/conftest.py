import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.pp import knn, preprocessing, wnn
from atlas.tl import PalantirExtension

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
_KNN, _N_PCS, _N_NEIGHBORS, _N_COMPS = 5, 5, 5, 10


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
        n_multineighbors=30,
        n_bandwidth_neighbors=_KNN,
        random_state=_SEED,
    )
    mudata.obs["pseudotime"] = np.linspace(0, 1, len(_CELLS))
    return mudata, "wnn"


def _via_wnn() -> tuple[MuData, str]:
    mudata = _base()
    wnn(mudata, knn=_KNN, n_pcs=_N_PCS, n_neighbors=_N_NEIGHBORS, random_state=_SEED)
    return mudata, "wnn"


def _via_knn() -> tuple[MuData, str]:
    mudata = _base(with_activity=False)
    mudata.obsm["X_joint"] = np.random.default_rng(_SEED).random((len(_CELLS), 10))
    knn(mudata, use_rep="X_joint", n_neighbors=_N_NEIGHBORS, random_state=_SEED)
    return mudata, "joint"


def _via_custom_key() -> tuple[MuData, str]:
    mudata = _base()
    wnn(mudata, knn=_KNN, n_pcs=_N_PCS, n_neighbors=_N_NEIGHBORS, key_added="graph", random_state=_SEED)
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


# --------------------------------------------------------------------------------------
# state representation
#
# The objects the state helpers are tested against. Moved here from
# `test_state_representation.py` when that file was split per function: nine modules now
# need them.
#
# Function-scoped and uncached on purpose. `migrate_states` and `reset_state_colors`
# mutate their argument in place, so every test needs its own object; a Palantir run on
# this fixture costs 0.03-0.08s, which is not worth sharing to avoid.
# --------------------------------------------------------------------------------------

_STATE_SEED, _STATE_K, _STATE_WAYPOINTS = 42, 10, 20
_STATE_CELLS = [f"cell{i}" for i in range(100)]
_STATE_GENES = [f"gene{i}" for i in range(20)]
_STATE_ACT_VAR = [f"act{i}" for i in range(15)]
_STATE_EARLY_CELL = "cell0"


def _bare_mudata(n: int = 12) -> MuData:
    names = [f"c{i}" for i in range(n)]
    rna = AnnData(np.zeros((n, 2), dtype=np.float32))
    rna.obs_names = names
    return MuData({"rna": rna})


def _recorded_mudata() -> MuData:
    """States written directly, so the colour rules can be tested without an inference.

    ``t1`` is deliberately both an initial and a terminal state: that is the case the
    per-kind colour convention cannot handle on its own.
    """
    mudata = _bare_mudata()
    # c2 and c3 are in `t1` under *both* kinds, which `allow_overlap` permits and a single
    # column could not represent.
    mudata.obs["initial_states"] = pd.Categorical(["HSC"] * 2 + ["t1"] * 2 + [None] * 8)
    mudata.obs["terminal_states"] = pd.Categorical([None] * 2 + ["t1"] * 2 + [None] * 2 + ["Ery"] * 3 + [None] * 3)
    mudata.obs["macrostates"] = pd.Categorical(["HSC"] * 2 + ["t1"] * 4 + ["Ery"] * 3 + ["TAC"] * 3)
    return mudata


def _legacy_mudata() -> MuData:
    """An object as a version before this layout wrote it."""
    mudata = _bare_mudata()
    mudata.uns["initial_states"] = {"HSC": ["c0", "c1"]}
    mudata.uns["terminal_states"] = {"Ery": ["c8"], "Mye": ["c9"]}
    mudata.uns["intermediate_states"] = {"TAC": ["c4", "c5"]}
    mudata.uns["fate_state_colors"] = {
        "HSC": "#e41a1c",
        "Ery": "#377eb8",
        "Mye": "#4daf4a",
        "TAC": "#984ea3",
    }
    return mudata


def _palantir_mudata() -> MuData:
    rng = np.random.default_rng(_STATE_SEED)
    rna = AnnData(rng.random((len(_STATE_CELLS), len(_STATE_GENES))))
    act = AnnData(rng.random((len(_STATE_CELLS), len(_STATE_ACT_VAR))))
    rna.obs_names, act.obs_names = _STATE_CELLS, _STATE_CELLS
    rna.var_names, act.var_names = _STATE_GENES, _STATE_ACT_VAR

    mudata = MuData({"rna": rna, "activity": act})
    mudata.obs = pd.DataFrame(
        {"cluster": ["red" if i % 2 == 0 else "blue" for i in range(len(_STATE_CELLS))]},
        index=_STATE_CELLS,
    )

    n = len(_STATE_CELLS)
    rows = np.repeat(np.arange(n), _STATE_K)
    cols = rng.integers(0, n, size=n * _STATE_K)
    dists = rng.random(n * _STATE_K)
    mudata.obsp["wnn_distances"] = csr_matrix((dists, (rows, cols)), shape=(n, n))
    mudata.obsp["wnn_connectivities"] = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(n, n))
    mudata.uns["wnn"] = {"params": {"n_neighbors": _STATE_K}}
    mudata.obsm["multiscale"] = pd.DataFrame(np.random.default_rng(_STATE_SEED).random((n, 5)), index=_STATE_CELLS)
    mudata.obsm["eigenvectors"] = pd.DataFrame(np.random.default_rng(_STATE_SEED).random((n, 5)), index=_STATE_CELLS)
    return mudata


def _run_palantir(cluster_key: str | None) -> MuData:
    extension = PalantirExtension(_palantir_mudata())
    extension.run(
        early_cell=_STATE_EARLY_CELL,
        knn=10,
        cluster_key=cluster_key,
        num_waypoints=_STATE_WAYPOINTS,
        eigvec_key="eigenvectors",
        eigvec_multi_key="multiscale",
    )
    return extension.mudata


@pytest.fixture
def bare_mudata():
    """Factory for an object carrying no state record at all."""
    return _bare_mudata


@pytest.fixture
def recorded_mudata():
    """Factory for an object with states written directly, no inference run."""
    return _recorded_mudata


@pytest.fixture
def legacy_mudata():
    """Factory for an object in the layout a version before this one wrote."""
    return _legacy_mudata


@pytest.fixture
def palantir_run():
    """Factory running Palantir, taking the ``cluster_key`` to name states after.

    A factory rather than a value because several tests need two runs to compare, or a
    specific ``cluster_key`` rather than both.
    """
    return _run_palantir


@pytest.fixture(params=[None, "cluster"])
def palantir_mudata(request) -> MuData:
    """A Palantir result, parametrised over naming states after a cluster or not."""
    return _run_palantir(request.param)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------------------
# plotting
#
# The objects the plotting entry points are drawn from. Each is shared by the entry
# point's own module and by the module covering its superseded counterpart, which must
# be drawn from the same object to be comparable.
#
# Factories rather than values: several tests need two objects to compare, or a variant
# built with different arguments.
# --------------------------------------------------------------------------------------

_EMB_SEED, _EMB_CELLS = 42, 60
#: Carried by both modalities, so it can only be resolved by naming one of them.
_EMB_SHARED_FEATURE = "GATA1"


def _embedding_mudata() -> MuData:
    """Two modalities, two embeddings, and one `.obs` column of every kind that is drawn.

    The second embedding is deliberately one the wider ecosystem does not privilege by
    name: resolving it is the case the superseded entry point cannot do at all.
    """
    rng = np.random.default_rng(_EMB_SEED)
    obs_names = [f"cell{i}" for i in range(_EMB_CELLS)]

    rna = AnnData(rng.random((_EMB_CELLS, 4)).astype(np.float32))
    rna.obs_names = obs_names
    rna.var_names = [_EMB_SHARED_FEATURE, "rna_only", "g2", "g3"]

    activity = AnnData(rng.random((_EMB_CELLS, 4)).astype(np.float32))
    activity.obs_names = obs_names
    activity.var_names = [_EMB_SHARED_FEATURE, "activity_only", "a2", "a3"]

    mudata = MuData({"rna": rna, "activity": activity})

    time = np.linspace(0.0, 1.0, _EMB_CELLS)
    mudata.obsm["X_umap"] = np.column_stack([time * np.cos(time * 6), time * np.sin(time * 6)])
    mudata.obsm["X_diffmap"] = rng.random((_EMB_CELLS, 4))

    mudata.obs["pseudotime"] = time
    mudata.obs["n_counts"] = rng.integers(0, 5000, _EMB_CELLS)
    mudata.obs["celltype"] = pd.Categorical(np.array(["HSC", "TAC", "Ery"])[rng.integers(0, 3, _EMB_CELLS)])
    mudata.obs["labels"] = np.array(["left", "right"])[rng.integers(0, 2, _EMB_CELLS)]
    mudata.obs["is_root"] = time < 0.3
    mudata.obs["with_missing"] = np.where(time < 0.5, np.nan, time)

    return mudata


_TRD_SEED, _TRD_N_CELLS = 42, 200
_TRD_CELLS = [f"cell{i}" for i in range(_TRD_N_CELLS)]
_TRD_TFS = ["GATA1", "SPI1"]
_TRD_GENES = ["KLF1", "HBB", "SLC4A1"]
_TRD_LINEAGES = ["Ery", "Mye"]


def _trends_mudata(fate: "pd.DataFrame | None" = None, colours: bool = True) -> MuData:
    from atlas.tl.utils import _assign_state_colors

    rng = np.random.default_rng(_TRD_SEED)
    time = np.linspace(0, 1, _TRD_N_CELLS)

    rna = AnnData(rng.random((_TRD_N_CELLS, len(_TRD_TFS))).astype(np.float32))
    rna.obs_names, rna.var_names = _TRD_CELLS, _TRD_TFS
    activity = AnnData(rng.random((_TRD_N_CELLS, len(_TRD_GENES))).astype(np.float32))
    activity.obs_names, activity.var_names = _TRD_CELLS, _TRD_GENES

    mudata = MuData({"rna": rna, "activity": activity})
    mudata.obs["pseudotime"] = time
    if fate is None:
        fate = pd.DataFrame(np.c_[time, 1 - time], index=_TRD_CELLS, columns=_TRD_LINEAGES)
    mudata.obsm["fate_probabilities"] = fate

    if colours and list(fate.columns):
        assignment = pd.Series(pd.NA, index=_TRD_CELLS, dtype=object)
        for offset, lineage in enumerate(fate.columns):
            assignment.iloc[offset * 10 : (offset + 1) * 10] = lineage
        mudata.obs["terminal_states"] = pd.Categorical(assignment, categories=list(fate.columns))
        _assign_state_colors(mudata)
    return mudata


_FT_SEED, _FT_N_CELLS = 42, 300
_FT_CELLS = [f"cell{i}" for i in range(_FT_N_CELLS)]
_FT_GENES = ["GATA1", "SPI1", "KLF1"]
_FT_FATES = ["Ery", "Mk", "Mono"]


def _fate_tree_probabilities(fates: list[str], seed: int = _FT_SEED) -> "pd.DataFrame":
    """Probabilities that commit further along pseudotime, so the tree actually branches."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, _FT_N_CELLS)
    concentration = np.full((_FT_N_CELLS, len(fates)), 0.3)
    concentration[np.arange(_FT_N_CELLS), rng.integers(0, len(fates), _FT_N_CELLS)] += 6 * time
    values = np.array([rng.dirichlet(row) for row in concentration])
    return pd.DataFrame(values, columns=fates, index=_FT_CELLS)


def _fate_tree_mudata(
    fates: "list[str] | None" = None,
    initial: "dict[str, list[str]] | None" = None,
    superseded: bool = False,
    colours: bool = True,
) -> MuData:
    """A object carrying exactly what `fate_tree` reads: probabilities, a time, an embedding."""
    fates = _FT_FATES if fates is None else fates
    rng = np.random.default_rng(_FT_SEED)

    rna = AnnData(rng.random((_FT_N_CELLS, len(_FT_GENES))).astype(np.float32))
    rna.obs_names, rna.var_names = _FT_CELLS, _FT_GENES
    mudata = MuData({"rna": rna})

    mudata.obs["pseudotime"] = np.linspace(0, 1, _FT_N_CELLS)
    mudata.obs["celltype"] = pd.Categorical(rng.choice(["prog", "inter", "mature"], _FT_N_CELLS))
    mudata.obsm["fate_probabilities"] = _fate_tree_probabilities(fates)
    mudata.obsm["X_umap"] = rng.normal(size=(_FT_N_CELLS, 2))

    initial = {"HSC": _FT_CELLS[:5]} if initial is None else initial
    if superseded:
        # The layout an object written by an earlier version carries.
        mudata.uns["initial_states"] = initial
    else:
        assignment = pd.Series(pd.NA, index=_FT_CELLS, dtype="object")
        for name, cells in initial.items():
            assignment.loc[cells] = name
        mudata.obs["initial_states"] = pd.Categorical(assignment)

    terminal = pd.Series(pd.NA, index=_FT_CELLS, dtype="object")
    for position, fate in enumerate(fates):
        terminal.iloc[-(position + 1) * 3 : len(_FT_CELLS) - position * 3 or None] = fate
    mudata.obs["terminal_states"] = pd.Categorical(terminal)
    if colours:
        categories = list(mudata.obs["terminal_states"].cat.categories)
        palette = dict(zip(_FT_FATES, ["#e41a1c", "#377eb8", "#4daf4a"], strict=False))
        mudata.uns["terminal_states_colors"] = [palette.get(name, "#999999") for name in categories]

    return mudata


@pytest.fixture
def embedding_mudata():
    """Factory for the object `atlas.pl.embedding` and `plot_embedding` are drawn from."""
    return _embedding_mudata


@pytest.fixture
def trends_mudata():
    """Factory for the object `atlas.pl.trends` and `plot_trends` are drawn from."""
    return _trends_mudata


@pytest.fixture
def fate_tree_mudata():
    """Factory for the object `atlas.pl.fate_tree` and `plot_tree` are drawn from."""
    return _fate_tree_mudata


@pytest.fixture
def fate_tree_probabilities():
    """Factory for the fate probabilities `fate_tree` is fitted to."""
    return _fate_tree_probabilities
