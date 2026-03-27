import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import terminal_state_silhouette

SEED = 42
N_CELLS, N_GENES = 100, 20
CELLS = [f"cell{i}" for i in range(N_CELLS)]


@pytest.fixture
def mudata():
    rng = np.random.default_rng(SEED)

    X_rna = rng.random((N_CELLS, N_GENES))
    X_act = rng.random((N_CELLS, N_GENES))

    rna = AnnData(X_rna, obs=pd.DataFrame(index=CELLS))
    act = AnnData(X_act, obs=pd.DataFrame(index=CELLS))

    mdata = MuData({"rna": rna, "activity": act})
    mdata.obs["pseudotime"] = np.linspace(0, 1, N_CELLS)

    return mdata


def make_fates(mudata, partial, columns):
    f = pd.DataFrame(0.0, index=mudata.obs_names, columns=columns)
    f.loc[partial.index] = partial
    return f


def test_no_fate_key(mudata):
    with pytest.raises(KeyError, match="not in mudata.obsm"):
        terminal_state_silhouette(mudata, fate_key="empty")


def test_no_fates(mudata):
    mudata.obsm["empty"] = pd.DataFrame(index=mudata.obs_names)

    with pytest.raises(ValueError, match="No terminal states"):
        terminal_state_silhouette(mudata, fate_key="empty")


def test_nan_in_fates(mudata):
    partial = pd.DataFrame([[1, 0], [np.nan, np.nan], [0, 1]], index=mudata.obs_names[:3], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    with pytest.raises(ValueError, match="NaN"):
        terminal_state_silhouette(mudata, fate_key="f")


def test_single_terminal(mudata):
    partial = pd.DataFrame({"A": [1, 1]}, index=mudata.obs_names[:2])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A"])

    with pytest.warns(UserWarning, match="Only one terminal state"):
        result = terminal_state_silhouette(mudata, fate_key="f")

    assert result == 0


def test_hard_requires_pseudotime(mudata):
    partial = pd.DataFrame([[1, 0], [0, 1]], index=mudata.obs_names[:2], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    with pytest.raises(KeyError):
        terminal_state_silhouette(mudata, fate_key="f", soft_assignment=False, time_key=None)


def test_soft_assignment_range(mudata):
    partial = pd.DataFrame([[1, 0], [0.8, 0.2], [0.2, 0.8], [0, 1]], index=mudata.obs_names[:4], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    S = terminal_state_silhouette(mudata, fate_key="f")
    assert -1 <= S <= 1


def test_hard_assignment_range(mudata):
    partial = pd.DataFrame([[1, 0], [1, 0], [0, 1], [0, 1]], index=mudata.obs_names[:4], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    S = terminal_state_silhouette(mudata, fate_key="f", soft_assignment=False, time_key="pseudotime")
    assert -1 <= S <= 1


def test_perfect_separation(mudata):
    partial = pd.DataFrame([[1, 0]] * 5 + [[0, 1]] * 5, index=mudata.obs_names[:10], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    S = terminal_state_silhouette(mudata, fate_key="f")
    assert S > 0.5


def test_no_separation(mudata):
    rng = np.random.default_rng(0)

    fates = pd.DataFrame(rng.random((len(mudata.obs_names), 2)), index=mudata.obs_names, columns=["A", "B"])
    fates = fates.div(fates.sum(axis=1), axis=0)

    mudata.obsm["f"] = fates

    S = terminal_state_silhouette(mudata, fate_key="f")
    assert abs(S) < 0.3


def test_no_nan_output(mudata):
    partial = pd.DataFrame([[1, 0], [1, 0]], index=mudata.obs_names[:2], columns=["A", "B"])
    mudata.obsm["f"] = make_fates(mudata, partial, ["A", "B"])

    S = terminal_state_silhouette(mudata, fate_key="f")
    assert not np.isnan(S)
