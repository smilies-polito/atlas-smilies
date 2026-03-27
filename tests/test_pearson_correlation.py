import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import pearson_correlation

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


def test_invalid_key1(mudata):
    with pytest.raises(KeyError, match=".obs"):
        pearson_correlation(mudata, key1="fake_key", key2="fake_key")


def test_invalid_key2(mudata):
    with pytest.raises(KeyError, match=".obs"):
        pearson_correlation(mudata, key1="pseudotime", key2="fake_key")


def test_nans(mudata):
    mudata.obs["entropy"] = np.nan
    with pytest.warns(UserWarning):
        stat, pval, ci = pearson_correlation(mudata=mudata, key1="pseudotime", key2="entropy")
    assert np.isnan(stat)
    assert np.isnan(pval)
    assert ci is None


def test_positive_correlation(mudata):
    mudata.obs["entropy"] = mudata.obs["pseudotime"].copy()
    stat, pval, ci = pearson_correlation(mudata=mudata, key1="pseudotime", key2="entropy")
    assert stat > 0.9
    assert pval < 0.05
    assert ci is not None
    low, high = ci
    assert low <= stat <= high or np.isnan(low)


def test_negative_correlation(mudata):
    mudata.obs["entropy"] = -mudata.obs["pseudotime"].copy()
    stat, pval, ci = pearson_correlation(mudata=mudata, key1="pseudotime", key2="entropy")
    assert stat < -0.9
    assert pval < 0.05
    assert ci is not None
    assert len(ci) == 2


def test_low_correlation(mudata):
    rng = np.random.default_rng(SEED)
    x = mudata.obs["pseudotime"].values
    noise = rng.normal(size=len(x))
    y = noise - np.dot(noise, x) / np.dot(x, x) * x
    mudata.obs["entropy"] = y
    stat, pval, ci = pearson_correlation(mudata=mudata, key1="pseudotime", key2="entropy")
    assert abs(stat) < 0.1
    assert pval > 0.05
    assert ci is not None


def test_bootstrap(mudata):
    rng = np.random.default_rng(SEED)
    subset_cells = mudata.obs.index[:200]
    mdata_sub = mudata[subset_cells].copy()
    pseudotime = mdata_sub.obs["pseudotime"].values
    noise = rng.normal(0, 0.05, len(pseudotime))
    mdata_sub.obs["entropy"] = pseudotime + noise
    with pytest.warns(UserWarning, match="Less than 500"):
        stat, pval, ci = pearson_correlation(mudata=mdata_sub, key1="pseudotime", key2="entropy")
    assert stat > 0.8
    assert pval < 0.05
    assert ci is not None
    assert len(ci) == 2
    assert ci[0] <= stat <= ci[1]
    assert (ci[1] - ci[0]) < 0.3
