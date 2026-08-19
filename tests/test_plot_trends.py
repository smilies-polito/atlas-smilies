"""The superseded entry point for drawing trends along a lineage.

Covers :func:`atlas.pl.plot_trends`, deprecated in 1.1.0 and removed in 2.0.0. Its
contract for 1.x is that it announces its supersession and still draws what it drew;
:func:`atlas.pl.trends` that supersedes it is covered in ``test_trends.py``.
"""

import matplotlib.pyplot as plt
import pytest

import atlas

#: Mirrors the lineages `conftest._trends_mudata` builds; a drift here fails loudly.
LINEAGES = ["Ery", "Mye"]


@pytest.fixture
def mudata(trends_mudata):
    return trends_mudata()


def test_the_superseded_entry_point_announces_its_supersession(trends_mudata):
    with pytest.warns(FutureWarning) as record:
        atlas.pl.plot_trends(trends_mudata(), ptf="GATA1", gene="KLF1")

    # `mudata` emits FutureWarnings of its own, so the record is searched rather than indexed
    messages = [str(warning.message) for warning in record]
    assert any("atlas.pl.trends" in message and "2.0.0" in message for message in messages)


def test_the_superseded_entry_point_still_draws_what_it_drew(trends_mudata):
    with pytest.warns(FutureWarning):
        atlas.pl.plot_trends(trends_mudata(), ptf="GATA1", gene="KLF1")

    figure = plt.gcf()
    assert len(figure.axes) == 2
    assert {line.get_label() for line in figure.axes[0].get_lines()} == set(LINEAGES)


# --------------------------------------------------------------------------------------
# the guards the docstring documents
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ptf": "absent", "gene": "KLF1"}, "TF absent not available"),
        ({"ptf": "GATA1", "gene": "absent"}, "Gene absent not available"),
    ],
)
def test_a_factor_or_gene_that_does_not_exist_is_named(trends_mudata, kwargs, message):
    with pytest.raises(KeyError, match=message):
        atlas.pl.plot_trends(trends_mudata(), **kwargs)


def test_a_time_that_is_not_recorded_is_named(trends_mudata):
    mudata = trends_mudata()
    del mudata.obs["pseudotime"]

    with pytest.raises(KeyError, match="pseudotime not in mudata.obs"):
        atlas.pl.plot_trends(mudata, ptf="GATA1", gene="KLF1")


def test_probabilities_that_are_not_recorded_are_named(trends_mudata):
    mudata = trends_mudata()
    del mudata.obsm["fate_probabilities"]

    with pytest.raises(KeyError, match="fate_probabilities not in mudata.obsm"):
        atlas.pl.plot_trends(mudata, ptf="GATA1", gene="KLF1")


# --------------------------------------------------------------------------------------
# choosing which lineages are drawn
# --------------------------------------------------------------------------------------


def test_one_lineage_may_be_given_as_a_name(trends_mudata):
    with pytest.warns(FutureWarning):
        atlas.pl.plot_trends(trends_mudata(), ptf="GATA1", gene="KLF1", branches="Ery")

    assert {line.get_label() for line in plt.gcf().axes[0].get_lines()} == {"Ery"}


def test_a_lineage_that_was_not_fitted_is_skipped_rather_than_raising(trends_mudata):
    with pytest.warns(FutureWarning):
        atlas.pl.plot_trends(trends_mudata(), ptf="GATA1", gene="KLF1", branches=["Ery", "absent"])

    assert {line.get_label() for line in plt.gcf().axes[0].get_lines()} == {"Ery"}


def test_saving_writes_a_figure_under_the_working_directory(trends_mudata, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.warns(FutureWarning):
        atlas.pl.plot_trends(trends_mudata(), ptf="GATA1", gene="KLF1", save="drawn")

    assert (tmp_path / "figures" / "trends_drawn.png").is_file()
