"""A cell belonging to states of two kinds.

Covers :func:`atlas.tl.utils._resolve_overlap`.
"""

import warnings

import pytest

from atlas.tl.utils import _resolve_overlap, _states_to_column


def test_the_coarse_states_take_the_initial_assignment_and_warn(bare_mudata):
    """`macrostates` admits one value per cell, so an overlap has to resolve. Initial takes
    precedence, following CellRank, and the caller is told which cells were affected."""
    initial = _states_to_column({"HSC": ["c0", "c1"]}, bare_mudata().obs_names)
    terminal = _states_to_column({"Ery": ["c1"], "Mye": ["c8"]}, bare_mudata().obs_names)

    with pytest.warns(UserWarning, match="both an initial and a terminal state"):
        macrostates = _resolve_overlap(initial, terminal)

    assert macrostates.loc["c1"] == "HSC"
    assert macrostates.loc["c8"] == "Mye"
    assert set(macrostates.cat.categories) == {"HSC", "Ery", "Mye"}


def test_no_warning_where_nothing_overlaps(bare_mudata):
    initial = _states_to_column({"HSC": ["c0"]}, bare_mudata().obs_names)
    terminal = _states_to_column({"Ery": ["c8"]}, bare_mudata().obs_names)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _resolve_overlap(initial, terminal)


def test_both_assignments_survive_the_overlap(recorded_mudata):
    """Resolving the single-valued view must not discard either kind's record of the cell."""
    mudata = recorded_mudata()
    assert mudata.obs["initial_states"].loc["c2"] == "t1"
    assert mudata.obs["terminal_states"].loc["c2"] == "t1"
