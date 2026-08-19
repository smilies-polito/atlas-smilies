import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import terminal_pseudotime_enrichment

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
