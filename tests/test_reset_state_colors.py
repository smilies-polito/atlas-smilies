"""Restoring the colours an inference assigned.

Covers :func:`atlas.tl.reset_state_colors`. See ``test_assign_state_colors.py`` for the
property these tests take part in.
"""

import numpy as np
import pytest
import scanpy as sc
from muon import MuData

from atlas.tl import reset_state_colors
from atlas.tl.utils import _assign_state_colors

#: The three kinds of state, and the colour list each carries.
KINDS = ("initial_states", "terminal_states", "macrostates")


SEED = 42


def _colour_of(mudata: MuData, kind: str, name: str) -> str:
    """The colour a kind's list gives a state, resolved by name rather than by position."""
    return list(mudata.uns[f"{kind}_colors"])[list(mudata.obs[kind].cat.categories).index(name)]


def test_plotting_overwrites_a_colour_list(recorded_mudata):
    """The premise of the repair: the lists are the ecosystem's keys, so it writes them."""
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    mudata.obsm["X_umap"] = np.random.default_rng(SEED).random((mudata.n_obs, 2))
    before = _colour_of(mudata, "terminal_states", "t1")

    sc.pl.embedding(mudata, basis="X_umap", color="terminal_states", palette="tab10", show=False)

    assert _colour_of(mudata, "terminal_states", "t1") != before


def test_plotting_leaves_the_record_untouched(recorded_mudata):
    """What the repair depends on: the record is not named as the convention's keys are."""
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    mudata.obsm["X_umap"] = np.random.default_rng(SEED).random((mudata.n_obs, 2))
    before = dict(mudata.uns["atlas_state_palette"])

    sc.pl.embedding(mudata, basis="X_umap", color="terminal_states", palette="tab10", show=False)

    assert mudata.uns["atlas_state_palette"] == before


def test_restoring_puts_back_exactly_what_was_assigned(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    expected = dict(mudata.uns["atlas_state_palette"])
    mudata.uns["terminal_states_colors"] = ["#000000", "#111111"]

    reset_state_colors(mudata)

    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert _colour_of(mudata, kind, name) == expected[name]


def test_restoring_recovers_every_list_at_once(recorded_mudata):
    """A reconciliation between the lists could not recover this: nothing would be left to
    reconcile from."""
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    expected = dict(mudata.uns["atlas_state_palette"])
    for kind in KINDS:
        mudata.uns[f"{kind}_colors"] = ["#000000"] * len(mudata.obs[kind].cat.categories)

    reset_state_colors(mudata)

    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert _colour_of(mudata, kind, name) == expected[name]


def test_restoring_without_a_record_says_so(recorded_mudata):
    mudata = recorded_mudata()
    with pytest.raises(KeyError, match="atlas_state_palette"):
        reset_state_colors(mudata)
