"""Colour lists written onto the object.

Covers :func:`atlas.tl.utils._write_state_colors`. See ``test_assign_state_colors.py``
for the property these tests take part in.
"""

from muon import MuData

from atlas.tl.utils import _assign_state_colors

#: The three kinds of state, and the colour list each carries.
KINDS = ("initial_states", "terminal_states", "macrostates")


def _colour_of(mudata: MuData, kind: str, name: str) -> str:
    """The colour a kind's list gives a state, resolved by name rather than by position."""
    return list(mudata.uns[f"{kind}_colors"])[list(mudata.obs[kind].cat.categories).index(name)]


def test_every_list_is_aligned_to_its_own_categories(recorded_mudata):
    """The record is ordered by name and the categories are not, so pairing the two orderings
    would misalign every list while still producing one of the right length."""
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    palette = mudata.uns["atlas_state_palette"]
    for kind in KINDS:
        categories = list(mudata.obs[kind].cat.categories)
        assert list(mudata.uns[f"{kind}_colors"]) == [palette[name] for name in categories]


def test_the_superseded_mapping_agrees_with_the_lists(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    legacy = mudata.uns["fate_state_colors"]
    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert legacy[name] == _colour_of(mudata, kind, name)


def test_every_fate_probability_column_carries_a_colour(palantir_run):
    """`atlas.pl` indexes the colour mapping by fate-probability column name, so a name applied
    to one side and not the other raises there. This change does not otherwise touch it."""
    for cluster_key in (None, "cluster"):
        mudata = palantir_run(cluster_key)
        for column in mudata.obsm["fate_probabilities"].columns:
            assert column in mudata.uns["fate_state_colors"]
            assert column in set(mudata.obs["terminal_states"].cat.categories)
