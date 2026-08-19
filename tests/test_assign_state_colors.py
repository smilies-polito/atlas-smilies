"""Colours assigned to states.

Covers :func:`atlas.tl.utils._assign_state_colors`.

These tests carry one part of a property that spans several functions: **a state name
has exactly one colour, whatever kind it is recorded under and whichever inference
produced it**. The rest of that property lives in ``test_write_state_colors.py`` (the
lists stay aligned to their own categories), ``test_reset_state_colors.py`` (plotting
may overwrite a list, and the assignment can be restored) and
``test_guard_palette_collision.py`` (the superseded mapping is not silently consumed).
A change to any one of them should be checked against the others.
"""

import pandas as pd
from muon import MuData

from atlas.tl.utils import _assign_state_colors


def _colour_of(mudata: MuData, kind: str, name: str) -> str:
    """The colour a kind's list gives a state, resolved by name rather than by position."""
    return list(mudata.uns[f"{kind}_colors"])[list(mudata.obs[kind].cat.categories).index(name)]


def test_a_name_under_two_kinds_has_one_colour(recorded_mudata):
    """Assigning per kind gives each list the start of the colour cycle, so an initial and an
    unrelated terminal state come out identical while one state under both comes out twice."""
    mudata = recorded_mudata()
    _assign_state_colors(mudata)

    assert _colour_of(mudata, "initial_states", "t1") == _colour_of(mudata, "terminal_states", "t1")
    assert _colour_of(mudata, "macrostates", "t1") == _colour_of(mudata, "initial_states", "t1")


def test_distinct_states_have_distinct_colours(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    palette = mudata.uns["atlas_state_palette"]
    assert len(set(palette.values())) == len(palette)


def test_unrelated_states_of_different_kinds_do_not_collide(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    assert _colour_of(mudata, "initial_states", "HSC") != _colour_of(mudata, "terminal_states", "Ery")


def test_a_repeated_inference_keeps_the_colours_it_already_gave(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    before = dict(mudata.uns["atlas_state_palette"])

    mudata.obs["terminal_states"] = pd.Categorical([None] * 4 + ["t1"] * 2 + ["Ery"] * 3 + ["new"] * 3)
    _assign_state_colors(mudata)

    after = mudata.uns["atlas_state_palette"]
    assert {k: v for k, v in after.items() if k in before} == before
    assert "new" in after


def test_the_assignment_does_not_depend_on_which_inference_ran(bare_mudata):
    """The same names must be coloured the same way however they were produced."""
    one, other = bare_mudata(), bare_mudata()
    one.obs["terminal_states"] = pd.Categorical(["Ery"] * 6 + ["Mye"] * 6)
    other.obs["terminal_states"] = pd.Categorical(["Mye"] * 6 + ["Ery"] * 6)
    _assign_state_colors(one)
    _assign_state_colors(other)

    assert one.uns["atlas_state_palette"] == other.uns["atlas_state_palette"]
