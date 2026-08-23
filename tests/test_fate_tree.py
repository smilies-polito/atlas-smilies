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

NODES = 30


def _probabilities(fates: list[str], seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, N_CELLS)
    concentration = np.full((N_CELLS, len(fates)), 0.3)
    concentration[np.arange(N_CELLS), rng.integers(0, len(fates), N_CELLS)] += 6 * time
    values = np.array([rng.dirichlet(row) for row in concentration])
    return pd.DataFrame(values, columns=fates, index=CELLS)


def _mudata(
    fates: list[str] | None = None,
    initial: dict[str, list[str]] | None = None,
    superseded: bool = False,
    colours: bool = True,
) -> MuData:
    fates = FATES if fates is None else fates
    rng = np.random.default_rng(SEED)

    rna = AnnData(rng.random((N_CELLS, len(GENES))).astype(np.float32))
    rna.obs_names, rna.var_names = CELLS, GENES
    mudata = MuData({"rna": rna})

    mudata.obs["pseudotime"] = np.linspace(0, 1, N_CELLS)
    mudata.obs["celltype"] = pd.Categorical(rng.choice(["prog", "inter", "mature"], N_CELLS))
    mudata.obsm["fate_probabilities"] = _probabilities(fates)
    mudata.obsm["X_umap"] = rng.normal(size=(N_CELLS, 2))

    initial = {"HSC": CELLS[:5]} if initial is None else initial
    if superseded:
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
def mudata() -> MuData:
    return _mudata()


@pytest.fixture
def tree(mudata: MuData) -> AnnData:
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


def test_both_views_are_drawn(mudata: MuData) -> None:
    axes = _draw(mudata, nodes=NODES)

    assert len(axes) == 2
    assert all(isinstance(ax, Axes) for ax in axes)
    assert axes[0].collections
    assert axes[1].collections or axes[1].lines


def test_one_colour_reaches_both_views(mudata: MuData, tree: AnnData) -> None:
    axes = _draw(mudata, tree=tree, color="celltype")

    figure = axes[0].figure
    assert len(figure.legends) == 1
    assert all(ax.get_legend() is None for ax in axes)

    labels = {text.get_text() for text in figure.legends[0].get_texts()}
    assert labels == set(mudata.obs["celltype"].cat.categories)


def test_a_continuous_colour_draws_one_colour_bar(mudata: MuData, tree: AnnData) -> None:
    axes = _draw(mudata, tree=tree, color="pseudotime")

    assert len(axes[0].figure.axes) == 3
    assert not axes[0].figure.legends


def test_supplied_axes_are_drawn_into(mudata: MuData, tree: AnnData) -> None:
    figure, supplied = plt.subplots(1, 2)

    axes = _draw(mudata, tree=tree, color="celltype", ax=supplied)

    assert axes[0] is supplied[0]
    assert axes[1] is supplied[1]
    assert axes[0].figure is figure


def test_the_dendrogram_honours_supplied_axes(mudata: MuData, tree: AnnData) -> None:
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


def test_the_fit_is_reproducible(mudata: MuData) -> None:
    first = _draw(mudata, nodes=NODES, return_tree=True, random_state=7)
    second = _draw(mudata, nodes=NODES, return_tree=True, random_state=7)

    assert np.array_equal(first.uns["graph"]["tips"], second.uns["graph"]["tips"])
    assert np.array_equal(first.uns["graph"]["forks"], second.uns["graph"]["forks"])
    assert first.uns["graph"]["root"] == second.uns["graph"]["root"]


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


def test_a_single_initial_state_needs_no_naming(mudata: MuData) -> None:
    fitted = _draw(mudata, nodes=NODES, return_tree=True)

    assert "root" in fitted.uns["graph"]


def test_several_initial_states_and_none_named_raises() -> None:
    mudata = _mudata(initial={"HSC": CELLS[:5], "MPP": CELLS[5:10]})

    with pytest.raises(ValueError, match="more than one initial state") as raised:
        _draw(mudata, nodes=NODES)

    assert "HSC" in str(raised.value)
    assert "MPP" in str(raised.value)


def test_a_named_initial_state_is_honoured() -> None:
    mudata = _mudata(initial={"HSC": CELLS[:5], "MPP": CELLS[5:10]})

    fitted = _draw(mudata, nodes=NODES, root="MPP", return_tree=True)

    assert "root" in fitted.uns["graph"]


def test_a_root_that_is_not_a_recorded_state_raises(mudata: MuData) -> None:
    with pytest.raises(KeyError, match="not among the recorded initial states"):
        _draw(mudata, nodes=NODES, root="nope")


def test_initial_cells_the_object_no_longer_holds_raise() -> None:
    mudata = _mudata(initial={"HSC": ["cell0", "gone1", "gone2"]}, superseded=True)

    with pytest.raises(KeyError, match="no longer holds"):
        _draw(mudata, nodes=NODES)


def test_superseded_initial_states_are_still_read() -> None:
    mudata = _mudata(superseded=True)

    with pytest.warns(FutureWarning, match="migrate_states"):
        atlas.pl.fate_tree(mudata, nodes=NODES, return_tree=True, show=False)


def test_two_fates_draw() -> None:
    """Below three fates scFates fits on `[P(fate0), pseudotime]` rather than the circular
    projection, so this is a distinct path and not the same one with a smaller input."""
    mudata = _mudata(fates=["Ery", "Mk"])

    axes = _draw(mudata, nodes=NODES)

    assert len(axes) == 2


def test_fewer_than_two_fates_raises(mudata: MuData) -> None:
    mudata.obsm["fate_probabilities"] = _probabilities(FATES)[["Ery"]]

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


def test_the_superseded_entry_point_is_deprecated(mudata: MuData) -> None:
    with pytest.warns(FutureWarning, match="plot_tree"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            atlas.pl.plot_tree(mudata, embedding_key="nope")


def test_fates_that_are_not_in_alphabetical_order_fit() -> None:
    mudata = _mudata(fates=["Mono", "Ery", "Zeta", "Mk", "Alpha"])
    fitted = _draw(mudata, nodes=NODES, return_tree=True)

    assert fitted.obsm["lineages_fwd"].shape[1] == 5
    assert list(fitted.obs["term_states_fwd"].cat.categories) == [
        "Mono",
        "Ery",
        "Zeta",
        "Mk",
        "Alpha",
    ]
    assert len(fitted.uns["graph"]["tips"]) >= 2


def test_the_fitting_route_is_fed_under_the_keys_it_reads(mudata: MuData, tree: AnnData) -> None:
    from atlas.pl._new_functions import _LINEAGE_KEY, _STATES_KEY

    assert (_LINEAGE_KEY, _STATES_KEY) == ("lineages_fwd", "term_states_fwd")
    assert _LINEAGE_KEY in tree.obsm
    assert _STATES_KEY in tree.obs
    assert f"{_STATES_KEY}_colors" in tree.uns


def test_the_probabilities_are_fed_as_an_array(tree: AnnData) -> None:
    """Where 1.1.0 put a `cellrank.Lineage`, which could be indexed by fate name."""
    probabilities = tree.obsm["lineages_fwd"]

    assert type(probabilities) is np.ndarray
    assert probabilities.dtype == np.float64


def _with_probabilities(mudata: MuData, mutate) -> MuData:
    frame = mudata.obsm["fate_probabilities"]
    values = frame.to_numpy(dtype=float).copy()
    mutate(values)
    mudata.obsm["fate_probabilities"] = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    return mudata


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("not a number", lambda v: v.__setitem__((5, 1), np.nan), "not finite.*Mk"),
        ("infinite", lambda v: v.__setitem__((9, 0), np.inf), "not finite.*Ery"),
        ("negative", lambda v: v.__setitem__((3, 2), -0.1), "negative.*Mono"),
        ("a dead fate", lambda v: v.__setitem__((slice(None), 1), 0.0), "Mk with zero"),
    ],
)
def test_probabilities_that_cannot_express_a_structure_raise(mudata: MuData, label: str, mutate, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _draw(_with_probabilities(mudata, mutate), nodes=NODES)


def test_the_returned_structure_is_what_the_fitting_produced(mudata: MuData, tree: AnnData) -> None:
    assert "lineages_fwd" in tree.obsm
    assert "X_fate_simplex_fwd" in tree.obsm
    assert "term_states_fwd" in tree.obs
    assert "lineages_fwd_kl_divergence" in tree.obs
    assert "term_states_fwd_colors" in tree.uns

    axes = _draw(mudata, tree=tree)
    assert len(axes) == 2


def test_the_fitting_route_restates_a_quantity_atlas_already_records(mudata: MuData) -> None:
    atlas.tl.compute_entropy(mudata, fate_probability_key="fate_probabilities")
    fitted = _draw(mudata, nodes=NODES, return_tree=True)

    np.testing.assert_allclose(
        fitted.obs["lineages_fwd_kl_divergence"].to_numpy(dtype=float),
        fitted.obs["kl_divergence"].to_numpy(dtype=float),
        atol=1e-12,
    )
