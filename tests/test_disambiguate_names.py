"""Naming states after an annotation of the cells.

Covers :func:`atlas.tl.utils._disambiguate_names`.
"""

import re

import numpy as np

from atlas.tl.utils import _disambiguate_names

#: The three kinds of state, and the colour list each carries.
KINDS = ("initial_states", "terminal_states", "macrostates")


def test_naming_states_after_a_cluster_only_renames_them(palantir_run):
    """`cluster_key` is documented as renaming inferred states. It must not also merge them,
    which would let a naming choice change how many fates are reported."""
    bare, named = palantir_run(None), palantir_run("cluster")

    assert bare.obsm["fate_probabilities"].shape == named.obsm["fate_probabilities"].shape
    np.testing.assert_allclose(
        bare.obsm["fate_probabilities"].to_numpy(),
        named.obsm["fate_probabilities"].to_numpy(),
    )
    assert list(bare.obsm["fate_probabilities"].columns) != list(named.obsm["fate_probabilities"].columns)


def test_states_claiming_one_cluster_are_disambiguated(palantir_run):
    named = palantir_run("cluster")
    names = list(named.uns["initial_states"]) + list(named.uns["terminal_states"])

    assert len(set(names)) == len(names)
    bases = [re.sub(r"_\d+$", "", name) for name in names]
    assert set(bases) <= {"red", "blue"}

    # A base several states claim must be suffixed on every one of them. Asserting only
    # that the base is a cluster name passes whether or not a suffix was ever applied,
    # which is what let this run without the numbering it is named for.
    for base in set(bases):
        claimed = [name for name, name_base in zip(names, bases, strict=True) if name_base == base]
        if len(claimed) > 1:
            assert all(re.fullmatch(rf"{base}_\d+", name) for name in claimed)


def test_disambiguation_is_repeatable(palantir_run):
    assert list(palantir_run("cluster").uns["terminal_states"]) == list(palantir_run("cluster").uns["terminal_states"])


def test_a_cluster_no_state_corresponds_to_is_not_a_category(palantir_run):
    """The annotation states are named after is a naming source, not a category set."""
    named = palantir_run("cluster")
    for kind in KINDS:
        for name in named.obs[kind].cat.categories:
            assert name in set(named.uns["initial_states"]) | set(named.uns["terminal_states"])


# --------------------------------------------------------------------------------------
# the naming itself
#
# Reached through `run` above, where the fixture happens to give every state the same
# cluster. These call it directly so that both sides of the rule are exercised.
# --------------------------------------------------------------------------------------


def test_a_name_claimed_by_one_state_is_left_alone():
    proposed = {"cell0": "red", "cell1": "blue"}
    assert _disambiguate_names(proposed) == proposed


def test_a_name_claimed_by_several_is_numbered_in_the_order_proposed():
    resolved = _disambiguate_names({"cell0": "red", "cell1": "blue", "cell2": "red"})
    assert resolved == {"cell0": "red_1", "cell1": "blue", "cell2": "red_2"}


def test_numbering_a_name_does_not_number_the_others():
    resolved = _disambiguate_names({"a": "red", "b": "red", "c": "blue", "d": "green"})
    assert resolved["c"] == "blue" and resolved["d"] == "green"
