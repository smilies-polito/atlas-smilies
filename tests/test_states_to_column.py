"""The column each kind of state is recorded in.

Covers :func:`atlas.tl.utils._states_to_column`.
"""

import pandas as pd


def test_a_cell_carries_the_name_of_the_state_it_belongs_to(recorded_mudata):
    mudata = recorded_mudata()
    assert mudata.obs["initial_states"].loc["c0"] == "HSC"
    assert mudata.obs["terminal_states"].loc["c6"] == "Ery"


def test_a_cell_in_no_state_of_a_kind_carries_no_value(recorded_mudata):
    mudata = recorded_mudata()
    assert pd.isna(mudata.obs["terminal_states"].loc["c0"])
    assert mudata.obs["terminal_states"].notna().sum() == 5


def test_each_kind_records_only_its_own_states(recorded_mudata):
    mudata = recorded_mudata()
    assert list(mudata.obs["initial_states"].cat.categories) == ["HSC", "t1"]
    assert list(mudata.obs["terminal_states"].cat.categories) == ["Ery", "t1"]
    assert "TAC" not in list(mudata.obs["initial_states"].cat.categories)


def test_the_kinds_are_independent(recorded_mudata):
    """A cell in an initial state is not thereby excluded from a terminal one."""
    mudata = recorded_mudata()
    both = mudata.obs["initial_states"].notna() & mudata.obs["terminal_states"].notna()
    assert both.sum() == 2
