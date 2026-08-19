import matplotlib
import muon as mu
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

import atlas
from atlas.tl import umap

matplotlib.use("Agg")

SEED, CELLS = 42, 40


def _representation_mudata(modalities: int = 1) -> MuData:
    """A graph over a single integrated representation, on an object with `modalities` mods.

    The number of modalities is a parameter because it must not influence which
    implementation is chosen: an object can carry several and still have its graph built
    from one integrated representation, which is the case this entry point exists to serve.
    """
    rng = np.random.default_rng(SEED)
    mods = {}
    for i in range(modalities):
        mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
        mod.obs_names = [f"cell{j}" for j in range(CELLS)]
        mods[f"mod{i}"] = mod
    mudata = MuData(mods)
    mudata.obsm["X_joint"] = rng.random((CELLS, 8))
    atlas.pp.knn(mudata, use_rep="X_joint", n_neighbors=5, random_state=SEED)
    return mudata


def _wnn_mudata(extra_modality: bool = False) -> MuData:
    """A weighted graph over two modalities, optionally on an object carrying a third."""
    rng = np.random.default_rng(SEED)
    names = ["rna", "activity"] + (["spare"] if extra_modality else [])
    mods = {}
    for name in names:
        mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
        mod.obs_names = [f"cell{j}" for j in range(CELLS)]
        mods[name] = mod
    mudata = MuData(mods)
    for name in names:
        sc.pp.pca(mudata[name], n_comps=5, random_state=SEED)
    atlas.pp.wnn(mudata, knn=5, n_pcs=5, n_neighbors=5, random_state=SEED)
    return mudata


@pytest.fixture
def spy(monkeypatch):
    """Record which of the two implementations was reached.

    The coordinates are not reproducible bit-for-bit even when seeded, so the route taken
    is asserted directly rather than inferred from the result.
    """
    calls = []

    def _record(name, original):
        def wrapper(*args, **kwargs):
            calls.append((name, kwargs))
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(mu.tl, "umap", _record("muon", mu.tl.umap))
    monkeypatch.setattr(sc.tl, "umap", _record("scanpy", sc.tl.umap))
    return calls


def _entered(spy: list) -> str:
    """The implementation this entry point delegated to.

    muon's own implementation delegates to scanpy's once it has prepared its temporary
    object, so only the first call says which route was taken.
    """
    return spy[0][0]


# --------------------------------------------------------------------------------------
# module placement
# --------------------------------------------------------------------------------------


def test_module_does_not_shadow_the_entry_point():
    """A module sharing a name with an exported function replaces it once imported, and
    the project's import sorting fixes the order in which that happens."""
    from atlas.tl import embedding

    assert callable(atlas.tl.umap)
    assert atlas.tl.umap is embedding.umap
    assert "umap" in atlas.tl.__all__


# --------------------------------------------------------------------------------------
# the two routes
# --------------------------------------------------------------------------------------


def test_weighted_graph_is_embedded(spy):
    mdata = _wnn_mudata()
    umap(mdata)
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)
    assert "umap" in mdata.uns
    assert _entered(spy) == "muon"


def test_representation_graph_is_embedded(spy):
    mdata = _representation_mudata()
    umap(mdata)
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)
    assert "umap" in mdata.uns
    assert _entered(spy) == "scanpy"


def test_modality_outside_the_graph_is_ignored():
    """muon iterates the modalities the record names, not every one the object carries."""
    mdata = _wnn_mudata(extra_modality=True)
    umap(mdata)
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)
    assert "spare" in mdata.mod


def test_modality_count_does_not_decide_the_route(spy):
    """Several modalities and a graph over one integrated representation: the count is not
    evidence of the route, and using it would send this to muon, which would fail."""
    mdata = _representation_mudata(modalities=3)
    umap(mdata)
    assert _entered(spy) == "scanpy"
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)


def test_result_is_stored_where_plotting_looks():
    """The plotting entry points prepend ``X_`` to the basis they are given, so the
    embedding has to sit under the prefixed name whichever route produced it."""
    for mdata in (_wnn_mudata(), _representation_mudata()):
        umap(mdata)
        assert "X_umap" in mdata.obsm

    # `atlas.pl.embedding` resolves its basis with or without the prefix, so either form
    # of its default names the key stored above.
    assert atlas.pl.embedding.__defaults__[0] in {"umap", "X_umap"}


def test_nothing_else_is_left_behind():
    """The single-view route computes on a temporary object; only the embedding and its
    record belong on the caller's."""
    mdata = _representation_mudata()
    before = set(mdata.obsm) | set(mdata.uns)
    umap(mdata)
    assert (set(mdata.obsm) | set(mdata.uns)) - before == {"X_umap", "umap"}


# --------------------------------------------------------------------------------------
# locating the graph
# --------------------------------------------------------------------------------------


def test_single_graph_is_found():
    mdata = _representation_mudata()
    umap(mdata)
    assert "X_umap" in mdata.obsm


def test_graph_under_a_non_default_name_is_found():
    """Detection keys on the provenance the routes stamp, not on the name they default to."""
    rng = np.random.default_rng(SEED)
    mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
    mod.obs_names = [f"cell{j}" for j in range(CELLS)]
    mdata = MuData({"mod0": mod})
    mdata.obsm["X_joint"] = rng.random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, key_added="whatever", random_state=SEED)

    umap(mdata)
    assert "X_umap" in mdata.obsm


def test_no_graph_names_the_entry_points_that_build_one():
    rng = np.random.default_rng(SEED)
    mod = AnnData(rng.random((CELLS, 12)).astype(np.float32))
    mdata = MuData({"mod0": mod})
    with pytest.raises(KeyError, match="atlas.pp.wnn"):
        umap(mdata)


def test_several_graphs_are_reported_rather_than_chosen_between():
    """An object carrying both was built deliberately; which one to embed is not inferable."""
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    with pytest.raises(ValueError, match="several") as excinfo:
        umap(mdata)
    assert "wnn" in str(excinfo.value)
    assert "joint" in str(excinfo.value)


def test_naming_a_graph_skips_detection():
    """Naming one resolves the ambiguity detection refuses to resolve on its own."""
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    umap(mdata, neighbors_key="joint")
    assert mdata.obsm["X_umap"].shape == (CELLS, 2)


def test_unknown_name_is_reported():
    mdata = _representation_mudata()
    with pytest.raises(KeyError, match="ghost"):
        umap(mdata, neighbors_key="ghost")


# --------------------------------------------------------------------------------------
# graphs this entry point does not cover
# --------------------------------------------------------------------------------------


def test_graph_without_provenance_is_refused_rather_than_inferred():
    """`atlas.pp.preprocessing` records nothing and embeds during its own run, so its
    output is out of scope; so is any graph built outside the package."""
    mdata = _representation_mudata()
    del mdata.uns["joint"]["atlas"]

    with pytest.raises(KeyError, match="no record of how it was built"):
        umap(mdata, neighbors_key="joint")


@pytest.mark.parametrize(
    ("route", "use_rep"),
    [("wnn", "X_joint"), ("representation", {"rna": -1, "activity": -1})],
)
def test_contradictory_record_fails_here(route, use_rep):
    """Delegating one of these fails inside muon or scanpy with a message naming neither
    the graph nor the fix."""
    mdata = _representation_mudata()
    mdata.uns["joint"]["atlas"]["route"] = route
    mdata.uns["joint"]["params"]["use_rep"] = use_rep

    with pytest.raises(ValueError, match="joint"):
        umap(mdata, neighbors_key="joint")


def test_missing_representation_is_reported():
    mdata = _representation_mudata()
    del mdata.obsm["X_joint"]
    with pytest.raises(KeyError, match="X_joint"):
        umap(mdata, neighbors_key="joint")


def test_missing_modality_is_reported():
    mdata = _wnn_mudata()
    mdata.uns["wnn"]["params"]["use_rep"] = {"rna": -1, "ghost": -1}
    with pytest.raises(KeyError, match="ghost"):
        umap(mdata, neighbors_key="wnn")


def test_a_mismatched_graph_cannot_reach_the_entry_point():
    """No dimension guard exists here because MuData will not hold a matrix that would
    trip one: it validates the shape at assignment, naming the key and both shapes."""
    from scipy.sparse import csr_matrix

    mdata = _representation_mudata()
    with pytest.raises(ValueError, match="incorrect shape"):
        mdata.obsp["joint_connectivities"] = csr_matrix((CELLS, CELLS - 1))


# --------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------


def test_copy_leaves_the_input_untouched():
    mdata = _representation_mudata()
    returned = umap(mdata, copy=True)
    assert "X_umap" in returned.obsm
    assert "X_umap" not in mdata.obsm


def test_without_copy_the_input_is_annotated():
    mdata = _representation_mudata()
    returned = umap(mdata, copy=False)
    assert returned is mdata
    assert "X_umap" in mdata.obsm


@pytest.mark.parametrize("build", [_representation_mudata, _wnn_mudata])
def test_embedding_parameters_are_forwarded(build, spy):
    """The signature mirrors the two implementations, so every parameter reaches whichever
    one is used, unchanged."""
    mdata = build()
    umap(mdata, n_components=3, min_dist=0.3, spread=1.5, random_state=7)

    _, forwarded = spy[0]
    assert forwarded["min_dist"] == 0.3
    assert forwarded["spread"] == 1.5
    assert forwarded["random_state"] == 7
    assert mdata.obsm["X_umap"].shape == (CELLS, 3)


def test_a_second_embedding_replaces_the_first():
    """The location is fixed, so this is documented rather than worked around."""
    mdata = _wnn_mudata()
    mdata.obsm["X_joint"] = np.random.default_rng(SEED).random((CELLS, 8))
    atlas.pp.knn(mdata, use_rep="X_joint", n_neighbors=5, random_state=SEED)

    umap(mdata, neighbors_key="wnn")
    first = mdata.obsm["X_umap"].copy()
    umap(mdata, neighbors_key="joint")

    assert mdata.obsm["X_umap"].shape == first.shape
    assert not np.array_equal(mdata.obsm["X_umap"], first)


def test_plotting_finds_the_embedding_without_being_told_where_it_is():
    mdata = _wnn_mudata()
    umap(mdata)
    mdata.obs["pseudotime"] = np.linspace(0, 1, CELLS)
    atlas.pl.embedding(mdata, color="pseudotime", show=False)


def test_a_graph_recording_a_route_this_package_does_not_build_is_refused():
    """A record can name a route neither producer writes; it is refused, not guessed at.

    The other tests all carry one of the two routes this package writes, so this branch
    had never been taken.
    """
    mudata = _wnn_mudata()
    mudata.uns["wnn"]["atlas"]["route"] = "invented"

    with pytest.raises(ValueError, match="records an unknown route 'invented'"):
        umap(mudata)
