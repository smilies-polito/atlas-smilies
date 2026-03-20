import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

import atlas
from atlas.tl import compute_entropy

fate_probability_key = "fate_probabilities"
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
rng = np.random.default_rng(42)


def _create_mudata(n_terminal_states: None | int = 3) -> MuData:

    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})

    if n_terminal_states is not None:
        if n_terminal_states == 0:
            mudata.obsm[fate_probability_key] = pd.DataFrame(index=CELLS)
        else:
            colnames = [f"terminal_{i}" for i in range(n_terminal_states)]
            mudata.obsm[fate_probability_key] = pd.DataFrame(
                rng.random((len(CELLS), n_terminal_states)), index=CELLS, columns=colnames
            )
    return mudata


def test_package_has_version():
    assert atlas.__version__ is not None


def test_missing_fates():
    mdata = _create_mudata(n_terminal_states=None)

    with pytest.raises(ValueError, match="not available"):
        compute_entropy(mudata=mdata, fate_probability_key=fate_probability_key)


def test_no_pandas_dataframe():
    mdata = _create_mudata(n_terminal_states=None)
    mdata.obsm[fate_probability_key] = rng.uniform(size=(len(CELLS), 3))

    with pytest.raises(ValueError, match="DataFrame"):
        compute_entropy(mudata=mdata, fate_probability_key=fate_probability_key)


def test_ok():
    mdata = _create_mudata(n_terminal_states=2)
    compute_entropy(mudata=mdata, fate_probability_key=fate_probability_key)
    assert "shannon_entropy" in mdata.obs.columns
    assert "kl_divergence" in mdata.obs.columns
    assert np.all((mdata.obs["shannon_entropy"] >= 0) & (mdata.obs["shannon_entropy"] <= 1))
    assert np.all((mdata.obs["kl_divergence"] >= 0) & (mdata.obs["kl_divergence"] <= 1))


def test_no_terminal_states():
    mdata = _create_mudata(n_terminal_states=0)
    with pytest.warns(UserWarning, match="No terminal states"):
        compute_entropy(mudata=mdata, fate_probability_key=fate_probability_key)
    assert np.all(np.isnan(mdata.obs["shannon_entropy"]))
    assert np.all(np.isnan(mdata.obs["kl_divergence"]))


def test_single_terminal():
    mdata = _create_mudata(n_terminal_states=1)
    compute_entropy(mudata=mdata, fate_probability_key=fate_probability_key)
    assert np.allclose(mdata.obs["shannon_entropy"], 0)
    assert np.allclose(mdata.obs["kl_divergence"], 0)
