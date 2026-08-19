"""The superseded entry point for drawing a principal tree.

Covers :func:`atlas.pl.plot_tree`, deprecated in 1.1.0 and removed in 2.0.0. Its
contract for 1.x is that it announces its supersession and still guards its inputs;
:func:`atlas.pl.fate_tree` that supersedes it is covered in ``test_fate_tree.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from muon import MuData

import atlas


@pytest.fixture
def mudata(fate_tree_mudata):
    return fate_tree_mudata()


def test_the_superseded_entry_point_is_deprecated(mudata: MuData) -> None:
    with pytest.warns(FutureWarning, match="plot_tree"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            atlas.pl.plot_tree(mudata, embedding_key="nope")


# --------------------------------------------------------------------------------------
# the guards the docstring documents
#
# Each is documented under `Warns` and returns without fitting. They are checked in
# order, so every test below has to satisfy the ones before the one it exercises.
# --------------------------------------------------------------------------------------


@pytest.fixture
def drawable(fate_tree_mudata):
    """An object carrying everything `plot_tree` reads, including the entropy it wants."""

    def build(**kwargs):
        mudata = fate_tree_mudata(**kwargs)
        mudata.obs["kl_divergence"] = np.linspace(0, 1, mudata.n_obs)
        return mudata

    return build


def test_probabilities_that_are_not_recorded_warn_and_fit_nothing(drawable):
    mudata = drawable()
    del mudata.obsm["fate_probabilities"]

    with pytest.warns(UserWarning, match="fate probabilities are not available"):
        assert atlas.pl.plot_tree(mudata) is None


def test_the_entropy_it_needs_is_named_when_absent(fate_tree_mudata):
    with pytest.warns(UserWarning, match="Entropy as KL-divergence required"):
        assert atlas.pl.plot_tree(fate_tree_mudata()) is None


def test_an_embedding_that_is_not_recorded_warns(drawable):
    with pytest.warns(UserWarning, match="X_absent not in mudata.obsm"):
        assert atlas.pl.plot_tree(drawable(), embedding_key="absent") is None


def test_a_colour_that_is_not_recorded_warns(drawable):
    with pytest.warns(UserWarning, match=r"Specified color \(absent\) not in mudata.obs"):
        assert atlas.pl.plot_tree(drawable(), color="absent") is None


def test_no_fates_at_all_warns(drawable):
    mudata = drawable()
    mudata.obsm["fate_probabilities"] = pd.DataFrame(index=mudata.obs_names)

    with pytest.warns(UserWarning, match="Fate probabilities are not available"):
        assert atlas.pl.plot_tree(mudata) is None


def test_a_single_fate_cannot_make_a_tree(drawable):
    mudata = drawable()
    mudata.obsm["fate_probabilities"] = mudata.obsm["fate_probabilities"][["Ery"]]

    with pytest.warns(UserWarning, match="only one fate has been found"):
        assert atlas.pl.plot_tree(mudata) is None
