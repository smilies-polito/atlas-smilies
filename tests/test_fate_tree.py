import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.axes import Axes
from muon import MuData

import atlas

matplotlib.use("Agg")

SEED = 42
N_CELLS = 300
CELLS = [f"cell{i}" for i in range(N_CELLS)]
GENES = ["GATA1", "SPI1", "KLF1"]
FATES = ["Ery", "Mk", "Mono"]

# Small enough that a fit is well under a second, large enough to branch.
NODES = 30


def fate_tree_mudata(
    fates: list[str] | None,
    initial: dict[str, list[str]] | None,
    superseded: bool,
    colours: bool,
    fate_tree_probabilities,
    fate_tree_mudata,
) -> MuData:
    """A object carrying exactly what `fate_tree` reads: probabilities, a time, an embedding."""
    fates = FATES if fates is None else fates
    rng = np.random.default_rng(SEED)

    rna = AnnData(rng.random((N_CELLS, len(GENES))).astype(np.float32))
    rna.obs_names, rna.var_names = CELLS, GENES
    mudata = MuData({"rna": rna})

    mudata.obs["pseudotime"] = np.linspace(0, 1, N_CELLS)
    mudata.obs["celltype"] = pd.Categorical(rng.choice(["prog", "inter", "mature"], N_CELLS))
    mudata.obsm["fate_probabilities"] = fate_tree_probabilities(fates)
    mudata.obsm["X_umap"] = rng.normal(size=(N_CELLS, 2))

    initial = {"HSC": CELLS[:5]} if initial is None else initial
    if superseded:
        # The layout an object written by an earlier version carries.
        mudata.uns["initial_states"] = initial
    else:
        assignment = pd.Series(pd.NA, index=CELLS, dtype="object")
        for name, cells in initial.items():
            assignment.loc[cells] = name
        mudata.obs["initial_states"] = pd.Categorical(assignment)

    terminal = pd.Series(pd.NA, index=CELLS, dtype="object")
    for position, fate in enumerate(fates):
        terminal.iloc[-(position + 1) * 3 : len(CELLS) - position * 3 or None] = fate
    mudata.obs["terminal_states"] = pd.Categorical(terminal)
    if colours:
        categories = list(mudata.obs["terminal_states"].cat.categories)
        palette = dict(zip(FATES, ["#e41a1c", "#377eb8", "#4daf4a"], strict=False))
        mudata.uns["terminal_states_colors"] = [palette.get(name, "#999999") for name in categories]

    return mudata


@pytest.fixture
def mudata(fate_tree_mudata) -> MuData:
    return fate_tree_mudata()


@pytest.fixture
def tree(mudata: MuData) -> AnnData:
    """One fit, shared by the tests that only need something to draw."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = atlas.pl.fate_tree(mudata, nodes=NODES, return_tree=True, show=False)
    plt.close("all")
    return fitted


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _draw(mudata: MuData, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return atlas.pl.fate_tree(mudata, show=False, **kwargs)


# --------------------------------------------------------------------------------------
# both views, drawn together
# --------------------------------------------------------------------------------------


def test_both_views_are_drawn(mudata: MuData) -> None:
    axes = _draw(mudata, nodes=NODES)

    assert len(axes) == 2
    assert all(isinstance(ax, Axes) for ax in axes)
    # The principal graph is drawn as line segments over the embedding; the dendrogram places
    # its cells and its segments. Neither view is empty.
    assert axes[0].collections
    assert axes[1].collections or axes[1].lines


def test_one_colour_reaches_both_views(mudata: MuData, tree: AnnData) -> None:
    axes = _draw(mudata, tree=tree, color="celltype")

    figure = axes[0].figure
    # One legend for the figure, not one per view.
    assert len(figure.legends) == 1
    assert all(ax.get_legend() is None for ax in axes)

    labels = {text.get_text() for text in figure.legends[0].get_texts()}
    assert labels == set(mudata.obs["celltype"].cat.categories)


def test_a_continuous_colour_draws_one_colour_bar(mudata: MuData, tree: AnnData) -> None:
    axes = _draw(mudata, tree=tree, color="pseudotime")

    # Two panels and one scale, rather than the same scale twice.
    assert len(axes[0].figure.axes) == 3
    assert not axes[0].figure.legends


def test_supplied_axes_are_drawn_into(mudata: MuData, tree: AnnData) -> None:
    figure, supplied = plt.subplots(1, 2)

    axes = _draw(mudata, tree=tree, color="celltype", ax=supplied)

    assert axes[0] is supplied[0]
    assert axes[1] is supplied[1]
    assert axes[0].figure is figure


def test_the_dendrogram_honours_supplied_axes(mudata: MuData, tree: AnnData) -> None:
    """`ax` is not a declared parameter of `scFates.pl.dendrogram`; it reaches
    `scanpy.pl.embedding` through `**kwargs`. Undocumented, so the most likely to drift."""
    _, supplied = plt.subplots(1, 2)

    _draw(mudata, tree=tree, ax=supplied)

    assert supplied[1].collections
    assert supplied[1].get_xlabel().startswith("dendro")


def test_other_than_two_axes_raises(mudata: MuData, tree: AnnData) -> None:
    _, supplied = plt.subplots(1, 3)

    with pytest.raises(ValueError, match="two axes"):
        _draw(mudata, tree=tree, ax=supplied[:1])

    with pytest.raises(ValueError, match="two axes"):
        _draw(mudata, tree=tree, ax=supplied)


# --------------------------------------------------------------------------------------
# the same object yields the same tree
# --------------------------------------------------------------------------------------


def test_the_fit_is_reproducible(mudata: MuData) -> None:
    """Two runs in one process on one machine, which is what the requirement claims. It does
    not claim reproducibility across platforms or `simpleppt` versions."""
    first = _draw(mudata, nodes=NODES, return_tree=True, random_state=7)
    second = _draw(mudata, nodes=NODES, return_tree=True, random_state=7)

    assert np.array_equal(first.uns["graph"]["tips"], second.uns["graph"]["tips"])
    assert np.array_equal(first.uns["graph"]["forks"], second.uns["graph"]["forks"])
    assert first.uns["graph"]["root"] == second.uns["graph"]["root"]


# --------------------------------------------------------------------------------------
# keeping and reusing a fitted tree
# --------------------------------------------------------------------------------------


def test_return_tree_hands_back_the_fit(mudata: MuData) -> None:
    fitted = _draw(mudata, nodes=NODES, return_tree=True)

    assert isinstance(fitted, AnnData)
    assert fitted.n_obs == mudata.n_obs
    for key in ("B", "F", "pp_info", "pp_seg", "tips", "forks", "root", "milestones"):
        assert key in fitted.uns["graph"]
    assert "X_R" in fitted.obsm
    assert "X_dendro" in fitted.obsm


def test_a_returned_tree_outlives_its_session(mudata: MuData, tree: AnnData, tmp_path) -> None:
    import anndata

    path = tmp_path / "tree.h5ad"
    tree.write_h5ad(path)
    restored = anndata.read_h5ad(path)

    assert isinstance(restored.uns["graph"]["pp_seg"], pd.DataFrame)
    assert isinstance(restored.uns["graph"]["pp_info"], pd.DataFrame)

    axes = _draw(mudata, tree=restored, color="celltype")
    assert len(axes) == 2


def test_supplying_a_tree_does_not_fit_again(mudata: MuData, tree: AnnData, monkeypatch) -> None:
    import scFates as scf

    def _fail(*args, **kwargs):
        raise AssertionError("the tree was fitted again")

    monkeypatch.setattr(scf.tl, "cellrank_to_tree", _fail)

    axes = _draw(mudata, tree=tree, color="celltype")
    assert len(axes) == 2


def test_a_colour_added_after_the_fit_is_drawn(mudata: MuData, tree: AnnData) -> None:
    """The ordinary path: fit, read the figure, annotate the branches it revealed, recolour."""
    assert "leiden" not in tree.obs.columns

    rng = np.random.default_rng(SEED)
    mudata.obs["leiden"] = pd.Categorical(rng.choice(["0", "1"], N_CELLS))

    axes = _draw(mudata, tree=tree, color="leiden")

    labels = {text.get_text() for text in axes[0].figure.legends[0].get_texts()}
    assert labels == {"0", "1"}


def test_a_tree_fitted_on_other_cells_raises(mudata: MuData, tree: AnnData) -> None:
    subset = mudata[: N_CELLS // 2].copy()

    with pytest.raises(ValueError, match="not the same cells"):
        _draw(subset, tree=tree)


def test_the_object_is_left_as_it_was_found(mudata: MuData) -> None:
    before = (sorted(mudata.obs.columns), sorted(mudata.obsm), sorted(mudata.uns))

    _draw(mudata, nodes=NODES, color="celltype")

    assert (sorted(mudata.obs.columns), sorted(mudata.obsm), sorted(mudata.uns)) == before


def test_the_object_is_left_as_it_was_found_when_reusing_a_tree(mudata: MuData, tree: AnnData) -> None:
    before = (sorted(mudata.obs.columns), sorted(mudata.obsm), sorted(mudata.uns))

    _draw(mudata, tree=tree, color="celltype")

    assert (sorted(mudata.obs.columns), sorted(mudata.obsm), sorted(mudata.uns)) == before


# --------------------------------------------------------------------------------------
# the root
# --------------------------------------------------------------------------------------


def test_a_single_initial_state_needs_no_naming(mudata: MuData) -> None:
    fitted = _draw(mudata, nodes=NODES, return_tree=True)

    assert "root" in fitted.uns["graph"]


def test_several_initial_states_and_none_named_raises(fate_tree_mudata) -> None:
    mudata = fate_tree_mudata(initial={"HSC": CELLS[:5], "MPP": CELLS[5:10]})

    with pytest.raises(ValueError, match="more than one initial state") as raised:
        _draw(mudata, nodes=NODES)

    assert "HSC" in str(raised.value)
    assert "MPP" in str(raised.value)


def test_a_named_initial_state_is_honoured(fate_tree_mudata) -> None:
    mudata = fate_tree_mudata(initial={"HSC": CELLS[:5], "MPP": CELLS[5:10]})

    fitted = _draw(mudata, nodes=NODES, root="MPP", return_tree=True)

    assert "root" in fitted.uns["graph"]


def test_a_root_that_is_not_a_recorded_state_raises(mudata: MuData) -> None:
    with pytest.raises(KeyError, match="not among the recorded initial states"):
        _draw(mudata, nodes=NODES, root="nope")


def test_initial_cells_the_object_no_longer_holds_raise(fate_tree_mudata) -> None:
    mudata = fate_tree_mudata(initial={"HSC": ["cell0", "gone1", "gone2"]}, superseded=True)

    with pytest.raises(KeyError, match="no longer holds"):
        _draw(mudata, nodes=NODES)


def test_superseded_initial_states_are_still_read(fate_tree_mudata) -> None:
    mudata = fate_tree_mudata(superseded=True)

    with pytest.warns(FutureWarning, match="migrate_states"):
        atlas.pl.fate_tree(mudata, nodes=NODES, return_tree=True, show=False)


# --------------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------------


def test_no_kl_divergence_is_required(mudata: MuData) -> None:
    """cellrank computes the priming degree from the fate probabilities; the column is never
    read. Requiring it made the caller compute something nothing consumes."""
    assert "kl_divergence" not in mudata.obs.columns

    axes = _draw(mudata, nodes=NODES)

    assert len(axes) == 2


def test_two_fates_draw(fate_tree_mudata) -> None:
    """Below three fates scFates fits on `[P(fate0), pseudotime]` rather than the circular
    projection, so this is a distinct path and not the same one with a smaller input."""
    mudata = fate_tree_mudata(fates=["Ery", "Mk"])

    axes = _draw(mudata, nodes=NODES)

    assert len(axes) == 2


def test_fewer_than_two_fates_raises(mudata: MuData, fate_tree_probabilities) -> None:
    mudata.obsm["fate_probabilities"] = fate_tree_probabilities(FATES)[["Ery"]]

    with pytest.raises(ValueError, match="at least two fates"):
        _draw(mudata, nodes=NODES)


def test_absent_probabilities_raise(mudata: MuData) -> None:
    del mudata.obsm["fate_probabilities"]

    with pytest.raises(KeyError, match="fate_probabilities"):
        _draw(mudata, nodes=NODES)


def test_an_unknown_basis_raises(mudata: MuData) -> None:
    with pytest.raises(KeyError, match="tsne"):
        _draw(mudata, nodes=NODES, basis="tsne")


def test_an_unknown_time_key_raises(mudata: MuData) -> None:
    with pytest.raises(KeyError, match="dpt"):
        _draw(mudata, nodes=NODES, time_key="dpt")


def test_an_unknown_colour_raises(mudata: MuData) -> None:
    with pytest.raises(KeyError, match="nope"):
        _draw(mudata, nodes=NODES, color="nope")


# --------------------------------------------------------------------------------------
# saving, showing, and the state the root is taken from
# --------------------------------------------------------------------------------------


def test_saving_writes_a_figure_under_the_working_directory(mudata, tree, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atlas.pl.fate_tree(mudata, tree=tree, save="drawn", show=False)

    assert (tmp_path / "figures" / "fate_tree_drawn.png").is_file()


def test_the_figure_is_shown_by_default_when_only_axes_come_back(mudata, tree, monkeypatch):
    """`show` defaults to the opposite of `return_tree`."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    atlas.pl.fate_tree(mudata, tree=tree)

    assert shown == [True]


def test_asking_for_the_tree_does_not_show_the_figure(mudata, tree, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    atlas.pl.fate_tree(mudata, tree=tree, return_tree=True)

    assert shown == []


def test_an_object_recording_no_initial_state_names_what_computes_one(fate_tree_mudata):
    """The root is read while fitting, so this is reached only when no tree is supplied."""
    mudata = fate_tree_mudata()
    del mudata.obs["initial_states"]

    with pytest.raises(KeyError, match="records no initial state"):
        atlas.pl.fate_tree(mudata, nodes=NODES, show=False)
