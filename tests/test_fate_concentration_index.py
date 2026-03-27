import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import fate_concentration_index

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

    fully_committed = pd.DataFrame({"A": np.zeros(N_CELLS), "B": np.ones(N_CELLS)}, index=CELLS)
    progressive = pd.DataFrame(
        [[0.33, 0.33, 0.34] if t < 0.5 else [t, 1 - t, 0] for t in pseudotime], columns=["A", "B", "C"], index=CELLS
    )
    single_fate = pd.DataFrame({"A": np.ones(N_CELLS)}, index=CELLS)
    empty_fate = pd.DataFrame([], index=CELLS)

    mdata.obsm["fully_committed"] = fully_committed
    mdata.obsm["progressive"] = progressive
    mdata.obsm["single_fate"] = single_fate
    mdata.obsm["empty_fate"] = empty_fate

    return mdata


def test_invalid_fate_key(mudata):
    with pytest.raises(KeyError, match=".obsm"):
        fate_concentration_index(mudata, fate_key="fake_key", time_key="fake_key")


def test_invalid_time_key(mudata):
    with pytest.raises(KeyError, match=".obs"):
        fate_concentration_index(mudata, fate_key="progressive", time_key="fake_key")


def test_no_fates(mudata):
    with pytest.warns(UserWarning, match="NaNs"):
        stat, pval, ci, index = fate_concentration_index(mudata, fate_key="empty_fate", time_key="pseudotime")
    assert np.isnan(stat)
    assert np.isnan(pval)
    assert ci is None
    assert index is None


def test_progressive_positive_correlation(mudata):
    stat, pval, ci, index = fate_concentration_index(mudata, fate_key="progressive", time_key="pseudotime")

    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert isinstance(index, pd.Series)
    assert stat > 0


def test_fully_committed_constant_index(mudata):
    stat, pval, ci, index = fate_concentration_index(mudata, fate_key="fully_committed", time_key="pseudotime")
    assert np.isnan(stat) or abs(stat) < 1e-6


def test_single_fate_constant(mudata):
    stat, pval, ci, index = fate_concentration_index(mudata, fate_key="single_fate", time_key="pseudotime")
    assert np.isnan(stat) or abs(stat) < 1e-6


def test_output_length(mudata):
    _, _, _, index = fate_concentration_index(mudata, fate_key="progressive", time_key="pseudotime")
    assert len(index) == N_CELLS
    assert list(index.index) == CELLS


def test_ci_bounds_valid(mudata):
    stat, _, ci, _ = fate_concentration_index(mudata, fate_key="progressive", time_key="pseudotime")

    low, high = ci
    assert low <= high
    assert low <= stat <= high


def test_small_sample_triggers_permutation_warning(mudata):
    subset_cells = mudata.obs.index[:200]
    mdata_small = mudata[subset_cells].copy()

    with pytest.warns(UserWarning, match="Small sample size"):
        stat, pval, ci, index = fate_concentration_index(mdata_small, fate_key="progressive", time_key="pseudotime")

    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert isinstance(ci, tuple)
    assert len(index) == 200
