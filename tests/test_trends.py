import os
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
N_CELLS = 200
CELLS = [f"cell{i}" for i in range(N_CELLS)]
TFS = ["GATA1", "SPI1"]
GENES = ["KLF1", "HBB", "SLC4A1"]
LINEAGES = ["Ery", "Mye"]


def _curve_colours(axes: list[Axes]) -> dict[str, str]:
    return {line.get_label(): line.get_color() for line in axes[0].get_lines()}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------------------
# the panels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("genes", "expected"), [("KLF1", 2), (["KLF1", "HBB"], 3), (GENES, 4)])
def test_a_panel_for_the_factor_and_one_per_gene(genes, expected, trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", genes, show=False)
    assert len(axes) == expected


def test_every_panel_is_the_same_size(trends_mudata):
    """The pseudotime axis must be rendered at one width throughout, or positions in different
    panels cannot be compared — which is what the figure is read for."""
    axes = atlas.pl.trends(trends_mudata(), "GATA1", GENES, show=False)
    sizes = {tuple(np.round(ax.get_position().size, 6)) for ax in axes}
    assert len(sizes) == 1


def test_the_panels_are_not_linked_to_one_another(trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", GENES, show=False)

    assert len({id(ax) for ax in axes}) == len(axes)
    for position, ax in enumerate(axes):
        for other in axes[position + 1 :]:
            assert not ax.get_shared_x_axes().joined(ax, other)
            assert not ax.get_shared_y_axes().joined(ax, other)


def test_each_panel_scales_its_own_vertical_axis(trends_mudata):
    """Expression and activity are different quantities."""
    axes = atlas.pl.trends(trends_mudata(), "GATA1", GENES, show=False)
    limits = {ax.get_ylim() for ax in axes}
    assert len(limits) > 1


def test_one_column_stacks_the_panels(trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", ["KLF1", "HBB"], ncols=1, show=False)
    lefts = {round(ax.get_position().x0, 6) for ax in axes}
    assert len(lefts) == 1


def test_unused_positions_are_removed_rather_than_drawn_empty(trends_mudata):
    """Three genes over two columns leaves a fourth cell; nothing empty may survive."""
    axes = atlas.pl.trends(trends_mudata(), "GATA1", ["KLF1", "HBB"], ncols=2, show=False)
    assert len(axes) == 3
    assert len(axes[0].figure.axes) == 3


# --------------------------------------------------------------------------------------
# lineage colours
# --------------------------------------------------------------------------------------


def test_lineages_are_drawn_in_the_colours_the_object_records(trends_mudata):
    mudata = trends_mudata()
    axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    recorded = dict(
        zip(
            [str(name) for name in mudata.obs["terminal_states"].cat.categories],
            list(mudata.uns["terminal_states_colors"]),
            strict=True,
        )
    )
    assert _curve_colours(axes) == recorded


def test_an_object_in_the_superseded_layout_still_draws_and_says_so(trends_mudata):
    mudata = trends_mudata(colours=False)
    mudata.uns["fate_state_colors"] = {"Ery": "#e41a1c", "Mye": "#377eb8"}

    with pytest.warns(FutureWarning, match="migrate_states"):
        axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert _curve_colours(axes) == {"Ery": "#e41a1c", "Mye": "#377eb8"}


def test_a_lineage_with_no_recorded_colour_is_drawn_quietly(trends_mudata):
    """The default is used without comment, as the entry point this replaces does."""
    mudata = trends_mudata(colours=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert set(_curve_colours(axes).values()) == {"grey"}


# --------------------------------------------------------------------------------------
# having nothing to draw, and the three reasons for it
# --------------------------------------------------------------------------------------


def test_an_object_recording_no_lineage_says_so(trends_mudata):
    mudata = trends_mudata(fate=pd.DataFrame(index=CELLS), colours=False)
    with pytest.raises(ValueError, match="records no lineage"):
        atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)


def test_a_lineage_that_does_not_exist_is_named_along_with_those_that_do(trends_mudata):
    """A mistyped name must not look like a lineage carrying no data."""
    with pytest.raises(ValueError, match="no lineage called 'typo'") as excinfo:
        atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", lineages="typo", show=False)

    message = str(excinfo.value)
    assert "'Ery'" in message and "'Mye'" in message


def test_every_lineage_skipped_is_reported_apart_from_there_being_none(trends_mudata):
    fate = pd.DataFrame(np.zeros((N_CELLS, 2)), index=CELLS, columns=LINEAGES)
    mudata = trends_mudata(fate=fate, colours=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="was skipped") as excinfo:
            atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert "records no lineage" not in str(excinfo.value)


def test_a_subset_of_lineages_can_be_drawn(trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", lineages="Ery", show=False)
    assert list(_curve_colours(axes)) == ["Ery"]


# --------------------------------------------------------------------------------------
# what comes back, and what is written
# --------------------------------------------------------------------------------------


def test_the_axes_come_back_by_default(trends_mudata):
    result = atlas.pl.trends(trends_mudata(), "GATA1", GENES, show=False)
    assert isinstance(result, list)
    assert all(isinstance(ax, Axes) for ax in result)


def test_the_models_come_back_when_asked_for(trends_mudata):
    result = atlas.pl.trends(trends_mudata(), "GATA1", GENES, return_models=True, show=False)

    assert sorted(result) == sorted(LINEAGES)
    for fitted in result.values():
        assert sorted(fitted) == ["gam_exp", "gams_act", "weights"]
        assert sorted(fitted["gams_act"]) == sorted(GENES)


def test_saving_happens_whether_or_not_the_models_are_asked_for(tmp_path, monkeypatch, trends_mudata):
    """CellRank returns before saving when the figure is wanted; that is not copied."""
    monkeypatch.chdir(tmp_path)

    atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", save="drawn", show=False)
    assert os.path.exists(tmp_path / "figures" / "trends_drawn.png")

    atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", save="models", return_models=True, show=False)
    assert os.path.exists(tmp_path / "figures" / "trends_models.png")


# --------------------------------------------------------------------------------------
# the superseded entry point
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# the modalities the figure is drawn from
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("present", ["rna", "activity"])
def test_a_missing_modality_is_named(present):
    """The factor is read from `rna` and the genes from `activity`; without both there is
    nothing to draw, and saying which is absent beats failing inside the fitting."""
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 1, N_CELLS)

    only = AnnData(rng.random((N_CELLS, 2)).astype(np.float32))
    only.obs_names = CELLS
    only.var_names = TFS if present == "rna" else GENES[:2]

    mudata = MuData({present: only})
    mudata.obs["pseudotime"] = time
    mudata.obsm["fate_probabilities"] = pd.DataFrame(np.c_[time, 1 - time], index=CELLS, columns=LINEAGES)

    absent = "activity" if present == "rna" else "rna"
    with pytest.raises(KeyError, match=absent):
        atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)


def test_both_modalities_present_draws(trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", show=False)
    assert len(axes) == 2


# --------------------------------------------------------------------------------------
# the remaining arguments
# --------------------------------------------------------------------------------------


def test_probabilities_that_are_not_recorded_are_named(trends_mudata):
    mudata = trends_mudata()
    del mudata.obsm["fate_probabilities"]

    with pytest.raises(KeyError, match="fate_probabilities not in mudata.obsm"):
        atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)


def test_a_size_can_be_given_rather_than_derived(trends_mudata):
    axes = atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", figsize=(9.0, 4.0), show=False)

    assert tuple(axes[0].get_figure().get_size_inches()) == (9.0, 4.0)


def test_the_figure_is_shown_by_default_when_nothing_is_returned(trends_mudata, monkeypatch):
    """`show` defaults to the opposite of `return_models`, so the axes path draws."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    atlas.pl.trends(trends_mudata(), "GATA1", "KLF1")

    assert shown == [True]


def test_asking_for_the_models_does_not_show_the_figure(trends_mudata, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    atlas.pl.trends(trends_mudata(), "GATA1", "KLF1", return_models=True)

    assert shown == []
