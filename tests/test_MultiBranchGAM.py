import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.pl import MultiBranchGAM

N_CELLS = 100
CELLS = [f"cell{i}" for i in range(N_CELLS)]


@pytest.fixture
def _create_mudata():

    pseudotime = np.linspace(0, 1, N_CELLS)
    fate = np.vstack([np.linspace(1, 0, N_CELLS), np.linspace(0, 1, N_CELLS)]).T

    fate_df = pd.DataFrame(fate, columns=["branch1", "branch2"], index=CELLS)

    X_rna = np.random.randn(N_CELLS, 1)
    rna = AnnData(X_rna, obs=pd.DataFrame([], index=CELLS))
    rna.var_names = ["ptf1"]

    X_act = np.random.randn(N_CELLS, 1)
    act = AnnData(X_act, obs=pd.DataFrame([], index=CELLS))
    act.var_names = ["gene1"]

    mdata = MuData({"rna": rna, "activity": act})
    mdata.obs["pseudotime"] = pseudotime
    mdata.obsm["fate_probabilities"] = fate_df

    return mdata


def test_init_valid(_create_mudata):
    model = MultiBranchGAM(_create_mudata, ptf="ptf1", gene="gene1")
    assert model.ptf == "ptf1"
    assert model.gene == "gene1"


def test_invalid_ptf(_create_mudata):
    with pytest.raises(KeyError):
        MultiBranchGAM(_create_mudata, ptf="wrong", gene="gene1")


def test_invalid_gene(_create_mudata):
    with pytest.raises(KeyError):
        MultiBranchGAM(_create_mudata, ptf="ptf1", gene="wrong")


def test_invalid_pseudotime(_create_mudata):
    _create_mudata.obs.drop(columns=["pseudotime"], inplace=True)
    with pytest.raises(KeyError):
        MultiBranchGAM(_create_mudata, ptf="ptf1", gene="gene1")


def test_fit(_create_mudata):
    model = MultiBranchGAM(_create_mudata, "ptf1", "gene1")
    model.fit()

    assert hasattr(model, "_models")
    assert len(model.models) == 2


def test_predict_shapes(_create_mudata):
    model = MultiBranchGAM(_create_mudata, "ptf1", "gene1")
    model.fit()
    model.predict(n_points=50)

    preds = model.predictions

    for _, res in preds.items():
        assert res["t_grid"].shape[0] == 50
        assert res["gex"].shape[0] == 50
        assert res["act"].shape[0] == 50


def test_confidence_intervals(_create_mudata):
    model = MultiBranchGAM(_create_mudata, "ptf1", "gene1")
    model.fit()
    model.predict(n_points=50)

    for _, res in model.predictions.items():
        assert np.all(res["gex_lower"] <= res["gex_upper"])
        assert np.all(res["act_lower"] <= res["act_upper"])


def test_zero_weights(_create_mudata):
    mdata = _create_mudata
    mdata.obsm["fate_probabilities"]["branch1"] = 0
    model = MultiBranchGAM(mdata, "ptf1", "gene1")

    with pytest.warns(UserWarning, match="transitioning"):
        model.fit()

    assert len(model.models) == 1
    assert "branch1" not in model.models
    assert "branch2" in model.models
