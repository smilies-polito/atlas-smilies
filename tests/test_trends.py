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
from atlas.tl.utils import _assign_state_colors

matplotlib.use("Agg")

SEED = 42
N_CELLS = 200
CELLS = [f"cell{i}" for i in range(N_CELLS)]
TFS = ["GATA1", "SPI1"]
GENES = ["KLF1", "HBB", "SLC4A1"]
LINEAGES = ["Ery", "Mye"]


def _mudata(fate: pd.DataFrame | None = None, colours: bool = True) -> MuData:
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 1, N_CELLS)

    rna = AnnData(rng.random((N_CELLS, len(TFS))).astype(np.float32))
    rna.obs_names, rna.var_names = CELLS, TFS
    activity = AnnData(rng.random((N_CELLS, len(GENES))).astype(np.float32))
    activity.obs_names, activity.var_names = CELLS, GENES

    mudata = MuData({"rna": rna, "activity": activity})
    mudata.obs["pseudotime"] = time
    if fate is None:
        fate = pd.DataFrame(np.c_[time, 1 - time], index=CELLS, columns=LINEAGES)
    mudata.obsm["fate_probabilities"] = fate

    if colours and list(fate.columns):
        assignment = pd.Series(pd.NA, index=CELLS, dtype=object)
        for offset, lineage in enumerate(fate.columns):
            assignment.iloc[offset * 10 : (offset + 1) * 10] = lineage
        mudata.obs["terminal_states"] = pd.Categorical(assignment, categories=list(fate.columns))
        _assign_state_colors(mudata)
    return mudata


def _curve_colours(axes: list[Axes]) -> dict[str, str]:
    return {line.get_label(): line.get_color() for line in axes[0].get_lines()}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(("genes", "expected"), [("KLF1", 2), (["KLF1", "HBB"], 3), (GENES, 4)])
def test_a_panel_for_the_factor_and_one_per_gene(genes, expected):
    axes = atlas.pl.trends(_mudata(), "GATA1", genes, show=False)
    assert len(axes) == expected


def test_every_panel_is_the_same_size():
    axes = atlas.pl.trends(_mudata(), "GATA1", GENES, show=False)
    sizes = {tuple(np.round(ax.get_position().size, 6)) for ax in axes}
    assert len(sizes) == 1


def test_the_panels_are_not_linked_to_one_another():
    axes = atlas.pl.trends(_mudata(), "GATA1", GENES, show=False)

    assert len({id(ax) for ax in axes}) == len(axes)
    for position, ax in enumerate(axes):
        for other in axes[position + 1 :]:
            assert not ax.get_shared_x_axes().joined(ax, other)
            assert not ax.get_shared_y_axes().joined(ax, other)


def test_each_panel_scales_its_own_vertical_axis():
    axes = atlas.pl.trends(_mudata(), "GATA1", GENES, show=False)
    limits = {ax.get_ylim() for ax in axes}
    assert len(limits) > 1


def test_one_column_stacks_the_panels():
    axes = atlas.pl.trends(_mudata(), "GATA1", ["KLF1", "HBB"], ncols=1, show=False)
    lefts = {round(ax.get_position().x0, 6) for ax in axes}
    assert len(lefts) == 1


def test_unused_positions_are_removed_rather_than_drawn_empty():
    axes = atlas.pl.trends(_mudata(), "GATA1", ["KLF1", "HBB"], ncols=2, show=False)
    assert len(axes) == 3
    assert len(axes[0].figure.axes) == 3


def test_lineages_are_drawn_in_the_colours_the_object_records():
    mudata = _mudata()
    axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    recorded = dict(
        zip(
            [str(name) for name in mudata.obs["terminal_states"].cat.categories],
            list(mudata.uns["terminal_states_colors"]),
            strict=True,
        )
    )
    assert _curve_colours(axes) == recorded


def test_an_object_in_the_superseded_layout_still_draws_and_says_so():
    mudata = _mudata(colours=False)
    mudata.uns["fate_state_colors"] = {"Ery": "#e41a1c", "Mye": "#377eb8"}

    with pytest.warns(FutureWarning, match="migrate_states"):
        axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert _curve_colours(axes) == {"Ery": "#e41a1c", "Mye": "#377eb8"}


def test_a_lineage_with_no_recorded_colour_is_drawn_quietly():
    mudata = _mudata(colours=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        axes = atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert set(_curve_colours(axes).values()) == {"grey"}


def test_an_object_recording_no_lineage_says_so():
    mudata = _mudata(fate=pd.DataFrame(index=CELLS), colours=False)
    with pytest.raises(ValueError, match="records no lineage"):
        atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)


def test_a_lineage_that_does_not_exist_is_named_along_with_those_that_do():
    with pytest.raises(ValueError, match="no lineage called 'typo'") as excinfo:
        atlas.pl.trends(_mudata(), "GATA1", "KLF1", lineages="typo", show=False)

    message = str(excinfo.value)
    assert "'Ery'" in message and "'Mye'" in message


def test_every_lineage_skipped_is_reported_apart_from_there_being_none():
    fate = pd.DataFrame(np.zeros((N_CELLS, 2)), index=CELLS, columns=LINEAGES)
    mudata = _mudata(fate=fate, colours=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="was skipped") as excinfo:
            atlas.pl.trends(mudata, "GATA1", "KLF1", show=False)

    assert "records no lineage" not in str(excinfo.value)


def test_a_subset_of_lineages_can_be_drawn():
    axes = atlas.pl.trends(_mudata(), "GATA1", "KLF1", lineages="Ery", show=False)
    assert list(_curve_colours(axes)) == ["Ery"]


def test_the_axes_come_back_by_default():
    result = atlas.pl.trends(_mudata(), "GATA1", GENES, show=False)
    assert isinstance(result, list)
    assert all(isinstance(ax, Axes) for ax in result)


def test_the_models_come_back_when_asked_for():
    result = atlas.pl.trends(_mudata(), "GATA1", GENES, return_models=True, show=False)

    assert sorted(result) == sorted(LINEAGES)
    for fitted in result.values():
        assert sorted(fitted) == ["gam_exp", "gams_act", "weights"]
        assert sorted(fitted["gams_act"]) == sorted(GENES)


def test_saving_happens_whether_or_not_the_models_are_asked_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atlas.pl.trends(_mudata(), "GATA1", "KLF1", save="drawn", show=False)
    assert os.path.exists(tmp_path / "figures" / "trends_drawn.png")

    atlas.pl.trends(_mudata(), "GATA1", "KLF1", save="models", return_models=True, show=False)
    assert os.path.exists(tmp_path / "figures" / "trends_models.png")


def test_the_superseded_entry_point_announces_its_supersession():
    with pytest.warns(FutureWarning) as record:
        atlas.pl.plot_trends(_mudata(), ptf="GATA1", gene="KLF1")

    messages = [str(warning.message) for warning in record]
    assert any("atlas.pl.trends" in message and "2.0.0" in message for message in messages)


@pytest.mark.parametrize("present", ["rna", "activity"])
def test_a_missing_modality_is_named(present):
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


def test_both_modalities_present_draws():
    axes = atlas.pl.trends(_mudata(), "GATA1", "KLF1", show=False)
    assert len(axes) == 2
