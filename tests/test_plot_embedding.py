"""The superseded entry point for drawing an embedding.

Covers :func:`atlas.pl.plot_embedding`, deprecated in 1.1.0 and removed in 2.0.0. Its
contract for 1.x is that it announces its supersession and still guards its inputs;
:func:`atlas.pl.embedding` that supersedes it is covered in ``test_embedding.py``.
"""

import matplotlib.pyplot as plt
import pytest

import atlas


@pytest.fixture
def mudata(embedding_mudata):
    return embedding_mudata()


def test_the_superseded_entry_point_announces_its_supersession(mudata):
    with pytest.warns(FutureWarning) as record:
        atlas.pl.plot_embedding(mudata, show=False)

    message = str(record[0].message)
    assert "atlas.pl.embedding" in message
    assert "2.0.0" in message


def test_the_superseded_entry_point_still_returns_nothing(mudata):
    with pytest.warns(FutureWarning):
        assert atlas.pl.plot_embedding(mudata, show=False) is None


def test_the_superseded_entry_point_still_leaves_the_object_alone(mudata):
    """It draws on a throwaway object; recording colours is confined to the replacement."""
    before = set(mudata.obsm), set(mudata.uns)

    with pytest.warns(FutureWarning):
        atlas.pl.plot_embedding(mudata, observation="celltype", show=False)

    assert (set(mudata.obsm), set(mudata.uns)) == before


# --------------------------------------------------------------------------------------
# the guards the docstring documents
#
# Both are documented under `Warns`, and both return without drawing.
# --------------------------------------------------------------------------------------


def test_an_observation_that_is_not_recorded_warns_and_draws_nothing(mudata):
    with pytest.warns(UserWarning, match="not a valid cell metadata"):
        assert atlas.pl.plot_embedding(mudata, observation="absent") is None

    assert not plt.get_fignums()


def test_an_embedding_that_is_not_recorded_warns_and_draws_nothing(mudata):
    with pytest.warns(UserWarning, match="not a valid cell embedding"):
        assert atlas.pl.plot_embedding(mudata, embedding_key="X_absent") is None

    assert not plt.get_fignums()
