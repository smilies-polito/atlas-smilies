import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import migrate_states, terminal_pseudotime_enrichment

SEED = 42
N_CELLS, N_GENES = 700, 40
CELLS = [f"cell{i}" for i in range(N_CELLS)]


@pytest.fixture
def mudata():

    pseudotime = np.linspace(0, 1, N_CELLS)
    rng = np.random.default_rng(SEED)

    X_rna = rng.random((N_CELLS, N_GENES))
    rna = AnnData(X_rna, obs=pd.DataFrame([], index=CELLS))

    X_act = rng.random((N_CELLS, N_GENES))
    act = AnnData(X_act, obs=pd.DataFrame([], index=CELLS))

    mdata = MuData({"rna": rna, "activity": act})
    mdata.obs["pseudotime"] = pseudotime

    return mdata


def _column(states: dict[str, list[str]]) -> pd.Categorical:
    """The states above, as the annotation that supersedes them."""
    assignment = pd.Series(pd.NA, index=CELLS, dtype=object)
    for name, cells in states.items():
        assignment.loc[list(cells)] = name
    return pd.Categorical(assignment, categories=list(states))


# --------------------------------------------------------------------------------------
# The cases below this point set only `uns["terminal_states"]`, the superseded layout, and
# so exercise the backward-compatibility path. They are kept exactly as they were: what
# they assert is what an object saved before the current layout must still score.
# Cases reading the annotation are grouped at the end of the module.
# --------------------------------------------------------------------------------------


def test_invalid_time_key(mudata):
    with pytest.raises(KeyError, match=".obs"):
        terminal_pseudotime_enrichment(mudata, time_key="fake_key")


def test_no_terminal_states(mudata):
    with pytest.warns(UserWarning):
        tep = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")
    assert np.isnan(tep)


def test_output_type(mudata):
    terminal_states = {"A": ["cell0", "cell1"], "B": ["cell2", "cell3"]}
    mudata.uns["terminal_states"] = terminal_states
    score = terminal_pseudotime_enrichment(mudata)
    assert isinstance(score, float)


def test_output_type_rank(mudata):
    terminal_states = {"A": ["cell0", "cell1"], "B": ["cell2", "cell3"]}
    mudata.uns["terminal_states"] = terminal_states
    score = terminal_pseudotime_enrichment(mudata, rank=True)
    assert isinstance(score, float)


def test_high_enrichment(mudata):
    terminal_states = {"A": mudata.obs_names[-10:], "B": mudata.obs_names[-20:-10]}
    mudata.uns["terminal_states"] = terminal_states
    score = terminal_pseudotime_enrichment(mudata)
    assert score > 0


def test_low_enrichment(mudata):
    terminal_states = {"A": mudata.obs_names[:10], "B": mudata.obs_names[20:30]}
    mudata.uns["terminal_states"] = terminal_states
    score = terminal_pseudotime_enrichment(mudata)
    assert score < 0


def test_single_cell(mudata):
    mdata_small = mudata[:1].copy()
    mdata_small.uns["terminal_states"] = {"only": mdata_small.obs.index}
    score = terminal_pseudotime_enrichment(mdata_small)
    assert isinstance(score, float)


def test_rank_single_cell(mudata):
    mdata_small = mudata[:1].copy()
    mdata_small.uns["terminal_states"] = {"only": mdata_small.obs.index}
    score = terminal_pseudotime_enrichment(mdata_small, rank=True)
    assert isinstance(score, float)


def test_rank_preserves_sign(mudata):
    mudata.uns["terminal_states"] = {"late": mudata.obs_names[-20:]}

    score_raw = terminal_pseudotime_enrichment(mudata, rank=False)
    score_rank = terminal_pseudotime_enrichment(mudata, rank=True)
    assert score_raw > 0
    assert score_rank > 0


def test_rank_invariant_to_scaling(mudata):
    mudata.uns["terminal_states"] = {"late": mudata.obs_names[-20:]}

    score1 = terminal_pseudotime_enrichment(mudata, rank=True)
    mudata_scaled = mudata.copy()
    mudata_scaled.obs["pseudotime"] *= 100
    score2 = terminal_pseudotime_enrichment(mudata_scaled, rank=True)
    assert score1 == pytest.approx(score2)


def test_rank_range(mudata):
    mudata.uns["terminal_states"] = {"mixed": mudata.obs_names[::10]}
    score = terminal_pseudotime_enrichment(mudata, rank=True)
    assert -1 <= score <= 1


# --------------------------------------------------------------------------------------
# reading the current layout
# --------------------------------------------------------------------------------------


def test_scores_from_the_annotation(mudata):
    states = {"A": CELLS[:10], "B": CELLS[-10:]}
    mudata.obs["terminal_states"] = _column(states)

    score = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    assert isinstance(score, float)
    assert not np.isnan(score)


@pytest.mark.parametrize("rank", [False, True])
def test_both_layouts_score_identically(mudata, rank):
    """The guarantee this change is built around: what the metric computes does not move."""
    states = {"A": CELLS[:10], "B": CELLS[-10:]}

    mudata.uns["terminal_states"] = states
    with pytest.warns(FutureWarning):
        from_superseded = terminal_pseudotime_enrichment(mudata, time_key="pseudotime", rank=rank)

    mudata.obs["terminal_states"] = _column(states)
    from_annotation = terminal_pseudotime_enrichment(mudata, time_key="pseudotime", rank=rank)

    assert from_annotation == from_superseded


def test_ordering_does_not_enter_the_result(mudata):
    """The two layouts enumerate states in different orders; the mean is over a set."""
    mudata.obs["terminal_states"] = _column({"A": CELLS[:10], "B": CELLS[-10:]})
    one = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    mudata.obs["terminal_states"] = _column({"B": CELLS[-10:], "A": CELLS[:10]})
    other = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    assert one == other


def test_the_superseded_layout_announces_itself(mudata):
    """A stored key cannot warn when it is read, so the function reading it warns instead."""
    mudata.uns["terminal_states"] = {"A": CELLS[:10]}

    with pytest.warns(FutureWarning) as record:
        terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    message = str(record[0].message)
    assert "migrate_states" in message
    assert "2.0.0" in message


def test_the_annotation_is_not_announced(mudata):
    mudata.obs["terminal_states"] = _column({"A": CELLS[:10]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        terminal_pseudotime_enrichment(mudata, time_key="pseudotime")


def test_an_empty_annotation_means_there_are_none(mudata):
    """Presence decides, not content: an annotation recording no state is an answer, and must
    not send the metric looking in the superseded record."""
    mudata.obs["terminal_states"] = pd.Categorical([None] * N_CELLS, categories=[])
    mudata.uns["terminal_states"] = {"A": CELLS[:10]}

    with pytest.warns(UserWarning, match="No terminal states"):
        score = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    assert np.isnan(score)


def test_migrating_does_not_move_the_score(mudata):
    """End to end: the same object, scored before and after conversion."""
    mudata.uns["terminal_states"] = {"A": CELLS[:10], "B": CELLS[-10:]}
    mudata.uns["initial_states"] = {"root": CELLS[:5]}
    mudata.uns["intermediate_states"] = {}

    with pytest.warns(FutureWarning):
        before = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    migrate_states(mudata)
    after = terminal_pseudotime_enrichment(mudata, time_key="pseudotime")

    assert after == before
