"""The states lying between the initial and terminal ones.

Covers :func:`atlas.tl.utils._intermediate_states`.
"""

import pandas as pd
import pytest

from atlas.tl.utils import _assign_state_colors, _intermediate_states


def test_the_coarse_states_are_recorded_whatever_produced_the_object(palantir_run):
    for mudata in (palantir_run(None), palantir_run("cluster")):
        assert "macrostates" in mudata.obs
        assert mudata.obs["macrostates"].notna().any()


def test_nothing_lies_between_states_where_the_coarse_states_are_their_union(palantir_run):
    assert _intermediate_states(palantir_run("cluster")) == {}


def test_intermediate_states_are_found_where_a_coarse_graining_exists(recorded_mudata):
    """Derived cell-wise, as the superseded `uns["intermediate_states"]` always was.

    A coarse state is designated initial or terminal through a subset of its cells — CellRank
    marks only the most representative — so its remaining cells are neither, and are reported
    here. `t1` appears despite being both an initial and a terminal state because only `c2` and
    `c3` carry those assignments, leaving `c4` and `c5`. This is what the previous
    implementation did, and preserving it is what keeps the deprecated key unchanged.
    """
    mudata = recorded_mudata()
    assert set(_intermediate_states(mudata)) == {"TAC", "t1"}
    assert _intermediate_states(mudata)["TAC"] == ["c9", "c10", "c11"]


def test_deriving_without_coarse_states_says_so(bare_mudata):
    mudata = bare_mudata()
    mudata.obs["terminal_states"] = pd.Categorical(["Ery"] * 12)
    with pytest.raises(KeyError, match="macrostates"):
        _intermediate_states(mudata)


def test_intermediate_states_are_not_a_kind_of_their_own(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    assert "intermediate_states" not in mudata.obs
    assert "intermediate_states_colors" not in mudata.uns
