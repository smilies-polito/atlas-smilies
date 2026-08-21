import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import to_hex
from muon import MuData

import atlas

matplotlib.use("Agg")

SEED = 42
N_CELLS = 200
CELLS = [f"cell{i}" for i in range(N_CELLS)]
GENES = ["GATA1", "SPI1", "KLF1"]
FATES = ["Ery", "Mk", "Neu"]
COLOURS = ["blue", "red", "green"]


def _mudata(fate: pd.DataFrame | None = None, colours: list[str] | None = None) -> MuData:
    """A trunk of contested cells splitting into committed branches."""
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((N_CELLS, len(GENES))).astype(np.float32))
    rna.obs_names, rna.var_names = CELLS, GENES
    mudata = MuData({"rna": rna})
    mudata.obsm["X_umap"] = rng.normal(size=(N_CELLS, 2))

    if fate is None:
        fate = pd.DataFrame(rng.dirichlet([0.4] * len(FATES), size=N_CELLS), index=CELLS, columns=FATES)
    mudata.obsm["fate_probabilities"] = fate

    names = list(fate.columns)
    if names:
        assignment = pd.Series(pd.NA, index=CELLS, dtype=object)
        for offset, fate_name in enumerate(names):
            assignment.iloc[offset * 10 : (offset + 1) * 10] = fate_name
        mudata.obs["terminal_states"] = pd.Categorical(assignment, categories=names)
        mudata.uns["terminal_states_colors"] = (colours or COLOURS)[: len(names)]
    return mudata


def _labels(ax) -> list[str]:
    """The fate names the figure carries, wherever they were placed."""
    legend = ax.get_legend()
    return sorted(
        [text.get_text() for text in ax.texts] + ([t.get_text() for t in legend.get_texts()] if legend else [])
    )


def _drawn(ax):
    """Facecolours of the layer carrying the cells, with the colormap resolved."""
    ax.figure.canvas.draw()
    layer = max(ax.collections, key=lambda c: len(c.get_offsets()))
    return [to_hex(c) for c in layer.get_facecolor()]


def _data_points(ax) -> int:
    """Points across the pair layers, past the background and the legend layer."""
    return sum(len(c.get_offsets()) for c in ax.collections[2:])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_every_cell_is_drawn_exactly_once():
    ax = atlas.pl.fate_probabilities(_mudata(), show=False)
    assert _data_points(ax) == N_CELLS


@pytest.mark.parametrize("n_fates", [2, 3, 4])
def test_every_cell_is_drawn_for_any_number_of_fates(n_fates):
    rng = np.random.default_rng(SEED)
    names = [f"F{i}" for i in range(n_fates)]
    fate = pd.DataFrame(rng.dirichlet([0.4] * n_fates, size=N_CELLS), index=CELLS, columns=names)
    ax = atlas.pl.fate_probabilities(_mudata(fate, ["blue", "red", "green", "orange"][:n_fates]), show=False)
    assert _data_points(ax) == N_CELLS


def test_each_fate_is_labelled():
    ax = atlas.pl.fate_probabilities(_mudata(), show=False)
    assert _labels(ax) == sorted(FATES)


def test_a_named_subset_draws_and_labels_only_those_fates():
    ax = atlas.pl.fate_probabilities(_mudata(), lineages=["Ery", "Mk"], show=False)
    assert _labels(ax) == ["Ery", "Mk"]


def test_the_names_sit_off_the_cells_by_default():
    ax = atlas.pl.fate_probabilities(_mudata(), show=False)
    assert [text.get_text() for text in ax.texts] == []
    assert ax.get_legend() is not None


def test_the_names_can_be_placed_on_the_cells():
    ax = atlas.pl.fate_probabilities(_mudata(), legend_loc="on data", show=False)
    assert sorted(text.get_text() for text in ax.texts) == sorted(FATES)
    assert ax.get_legend() is None


def test_the_names_can_be_suppressed():
    ax = atlas.pl.fate_probabilities(_mudata(), legend_loc="none", show=False)
    assert _labels(ax) == []


@pytest.mark.parametrize("loc", ["right margin", "on data", "best", "none"])
def test_every_placement_still_draws_every_cell(loc):
    ax = atlas.pl.fate_probabilities(_mudata(), legend_loc=loc, show=False)
    assert _data_points(ax) == N_CELLS


def test_one_recorded_fate_renders_flat_in_its_own_colour():
    fate = pd.DataFrame(np.ones((N_CELLS, 1)), index=CELLS, columns=["Ery"])
    ax = atlas.pl.fate_probabilities(_mudata(fate, ["blue"]), show=False)
    assert set(_drawn(ax)) == {to_hex("blue")}


def test_one_named_fate_of_several_is_shaded_not_flattened():
    ax = atlas.pl.fate_probabilities(_mudata(), lineages=["Ery"], show=False)
    assert len(set(_drawn(ax))) > 1


def test_a_single_recorded_fate_that_is_not_all_ones_is_shaded():
    fate = pd.DataFrame(np.linspace(0.0, 1.0, N_CELLS)[:, None], index=CELLS, columns=["Ery"])
    ax = atlas.pl.fate_probabilities(_mudata(fate, ["blue"]), show=False)
    assert len(set(_drawn(ax))) > 1


def test_no_fates_draws_every_cell_grey():
    fate = pd.DataFrame(index=CELLS)
    ax = atlas.pl.fate_probabilities(_mudata(fate), show=False)
    assert set(_drawn(ax)) == {to_hex("lightgrey")}


def test_no_probabilities_at_all_raises_naming_what_computes_them():
    mudata = _mudata()
    del mudata.obsm["fate_probabilities"]
    with pytest.raises(KeyError, match="fate_probabilities"):
        atlas.pl.fate_probabilities(mudata, show=False)


def test_a_fate_that_does_not_exist_raises_naming_what_is_available():
    with pytest.raises(KeyError, match="Ery"):
        atlas.pl.fate_probabilities(_mudata(), lineages=["Erythroid"], show=False)


def test_a_probability_that_is_not_a_number_warns_and_draws_nothing():
    rng = np.random.default_rng(SEED)
    values = rng.dirichlet([0.4] * len(FATES), size=N_CELLS)
    values[0, 0] = np.nan
    fate = pd.DataFrame(values, index=CELLS, columns=FATES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = atlas.pl.fate_probabilities(_mudata(fate), show=False)
    assert result is None
    assert any("not a number" in str(w.message) for w in caught)
    assert not plt.get_fignums()


def test_a_fate_with_no_recorded_colour_still_draws():
    mudata = _mudata()
    del mudata.uns["terminal_states_colors"]
    ax = atlas.pl.fate_probabilities(mudata, show=False)
    assert _data_points(ax) == N_CELLS


def test_the_object_is_left_as_it_was_found():
    mudata = _mudata()
    before_obs = list(mudata.obs.columns)
    before_labelled = int(mudata.obs["terminal_states"].notna().sum())
    atlas.pl.fate_probabilities(mudata, show=False)
    assert list(mudata.obs.columns) == before_obs
    assert int(mudata.obs["terminal_states"].notna().sum()) == before_labelled


def test_the_private_cellrank_path_is_not_bound_at_import_time():
    """The hazard is ATLAS referencing `cellrank._utils._lineage`, not the module existing."""
    import atlas.pl.plots as plots

    assert not hasattr(plots, "Lineage")
    assert not hasattr(plots, "scv")
    assert not hasattr(plots, "scf")


def _superseded_mudata() -> MuData:
    """The old entry point reads the superseded colour mapping, not the current list."""
    mudata = _mudata()
    mudata.uns["fate_state_colors"] = dict(zip(FATES, COLOURS, strict=True))
    return mudata


def test_the_superseded_entry_point_announces_its_supersession():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        atlas.pl.plot_fate_probabilities(_superseded_mudata(), show=False)
    messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert any("plot_fate_probabilities" in m and "2.0.0" in m for m in messages)
    assert any("atlas.pl.fate_probabilities" in m for m in messages)


def test_the_superseded_entry_point_still_draws():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas.pl.plot_fate_probabilities(_superseded_mudata(), show=False)
    assert plt.get_fignums()
