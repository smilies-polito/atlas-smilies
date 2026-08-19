"""The colour recorded for each state, from whichever layout records them.

Covers :func:`atlas.pl.utils._state_colors`, which the plotting entry points read so that
a lineage is drawn in the colour the object records for it.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.pl.utils import _state_colors

CELLS = [f"cell{i}" for i in range(6)]
KIND = "terminal_states"


def _object(states, colors=None, superseded=None) -> MuData:
    rna = AnnData(np.zeros((len(CELLS), 2), dtype=np.float32))
    rna.obs_names = CELLS
    mudata = MuData({"rna": rna})
    if states is not None:
        mudata.obs[KIND] = pd.Categorical(states)
    if colors is not None:
        mudata.uns[f"{KIND}_colors"] = list(colors)
    if superseded is not None:
        mudata.uns["fate_state_colors"] = dict(superseded)
    return mudata


def test_a_list_aligned_to_the_categories_is_paired_by_name():
    mudata = _object(["Ery"] * 3 + ["Mye"] * 3, ["#111111", "#222222"])

    assert _state_colors(mudata, KIND) == {"Ery": "#111111", "Mye": "#222222"}


def test_an_object_recording_nothing_gives_no_colour():
    assert _state_colors(_object(None), KIND) == {}


def test_the_superseded_mapping_is_read_when_no_list_is_recorded():
    mudata = _object(None, superseded={"Ery": "#111111"})

    with pytest.warns(FutureWarning, match="fate_state_colors"):
        assert _state_colors(mudata, KIND) == {"Ery": "#111111"}


def test_a_list_that_does_not_match_the_categories_falls_back_to_the_mapping():
    """Length is what decides. A list that cannot be paired by position is not guessed at.

    Reaching the superseded mapping while a list is *present* is the branch that had never
    been taken: every other case here records one or the other, not a disagreeing pair.
    """
    mudata = _object(
        ["Ery"] * 3 + ["Mye"] * 3,
        ["#111111", "#222222", "#333333"],  # three colours for two categories
        superseded={"Ery": "#aaaaaa", "Mye": "#bbbbbb"},
    )

    with pytest.warns(FutureWarning, match="fate_state_colors"):
        assert _state_colors(mudata, KIND) == {"Ery": "#aaaaaa", "Mye": "#bbbbbb"}
