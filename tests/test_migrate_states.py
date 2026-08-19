"""Converting an object written by an earlier version.

Covers :func:`atlas.tl.migrate_states`.
"""

import pytest
from muon import MuData

from atlas.tl import migrate_states

CELLS = [f"cell{i}" for i in range(100)]


def _colour_of(mudata: MuData, kind: str, name: str) -> str:
    """The colour a kind's list gives a state, resolved by name rather than by position."""
    return list(mudata.uns[f"{kind}_colors"])[list(mudata.obs[kind].cat.categories).index(name)]


def test_migrating_records_the_states_it_finds(legacy_mudata):
    mudata = legacy_mudata()
    migrate_states(mudata)

    assert list(mudata.obs["initial_states"].cat.categories) == ["HSC"]
    assert set(mudata.obs["terminal_states"].cat.categories) == {"Ery", "Mye"}
    assert mudata.obs["initial_states"].loc["c0"] == "HSC"


def test_migrating_reconstructs_the_coarse_states_as_the_union(legacy_mudata):
    mudata = legacy_mudata()
    migrate_states(mudata)
    assert set(mudata.obs["macrostates"].cat.categories) == {"HSC", "Ery", "Mye", "TAC"}
    assert mudata.obs["macrostates"].notna().sum() == 6


def test_migrating_carries_the_colours_over(legacy_mudata):
    mudata = legacy_mudata()
    migrate_states(mudata)
    assert mudata.uns["atlas_state_palette"]["HSC"] == "#e41a1c"
    assert _colour_of(mudata, "terminal_states", "Ery") == "#377eb8"


def test_migrating_leaves_what_was_there(legacy_mudata):
    mudata = legacy_mudata()
    migrate_states(mudata)
    for key in ("initial_states", "terminal_states", "intermediate_states", "fate_state_colors"):
        assert key in mudata.uns


def test_migrating_twice_changes_nothing(legacy_mudata):
    mudata = legacy_mudata()
    migrate_states(mudata)
    before = list(mudata.obs["macrostates"].astype(object)), dict(mudata.uns["atlas_state_palette"])

    migrate_states(mudata)

    assert (list(mudata.obs["macrostates"].astype(object)), dict(mudata.uns["atlas_state_palette"])) == before


def test_migrating_an_object_with_nothing_to_convert_says_so(bare_mudata):
    with pytest.raises(KeyError, match="nothing to convert"):
        migrate_states(bare_mudata())


def test_a_migrated_object_agrees_with_a_freshly_computed_one(bare_mudata, palantir_run):
    """Migration is exact: the union it reconstructs is what a fresh run writes."""
    fresh = palantir_run("cluster")
    legacy = bare_mudata(len(CELLS))
    legacy.obs_names = CELLS
    legacy.uns["initial_states"] = {k: list(v) for k, v in fresh.uns["initial_states"].items()}
    legacy.uns["terminal_states"] = {k: list(v) for k, v in fresh.uns["terminal_states"].items()}
    legacy.uns["intermediate_states"] = {}

    migrate_states(legacy)

    assert set(legacy.obs["macrostates"].cat.categories) == set(fresh.obs["macrostates"].cat.categories)
    assert list(legacy.obs["macrostates"].astype(object)) == list(fresh.obs["macrostates"].astype(object))
