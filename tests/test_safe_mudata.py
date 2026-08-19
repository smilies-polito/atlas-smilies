import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.pp import _safe_mudata

CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
PEAKS = [f"peak{i}" for i in range(50)]
SEED = 42


def _generate_activity() -> AnnData:
    rng = np.random.default_rng(SEED)
    activity = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    activity.obs_names = CELLS
    activity.var_names = ACTIVITY_VAR
    return activity


def _create_mudata(with_activity: bool = False, with_atac: bool = True, fragment_file: None | str = None) -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    rna.obs_names = CELLS
    rna.var_names = GENES

    mudata = MuData({"rna": rna})

    if with_atac:
        atac = AnnData(rng.random((len(CELLS), len(PEAKS))))
        atac.obs_names = CELLS
        atac.var_names = PEAKS
        mudata.mod["atac"] = atac
        if fragment_file is not None:
            mudata["atac"].uns["files"] = {}
            mudata["atac"].uns["files"]["fragments"] = fragment_file

    if with_activity:
        activity = _generate_activity()
        mudata.mod["activity"] = activity

    return mudata


def test_no_modalities():
    mudata = MuData({"dummy": AnnData(np.zeros((1, 1)))})
    with pytest.raises(KeyError):
        _ = _safe_mudata(mudata=mudata, modalities=["rna", "atac"])


def test_single_modality():
    mudata = _create_mudata(with_activity=False, with_atac=True, fragment_file=None)
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "rna" in new_data.mod
    assert "activity" not in new_data.mod
    assert "atac" not in new_data.mod


def test_keeps_all_modalities():
    mudata = _create_mudata(with_activity=True, with_atac=True, fragment_file=None)
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity", "atac"])
    assert "rna" in new_data.mod
    assert "activity" in new_data.mod
    assert "atac" in new_data.mod


def test_keeps_specified_modalities():
    mudata = _create_mudata(with_activity=True, with_atac=True, fragment_file=None)
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "rna" in new_data.mod
    assert "activity" in new_data.mod
    assert "atac" not in new_data.mod


def test_modality_obs():
    mudata = _create_mudata(with_activity=True, with_atac=True, fragment_file=None)
    mudata["rna"].obs["test"] = 2
    mudata.obs["test"] = 1
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "test" in new_data.obs
    assert "rna:test" in new_data.obs


def test_keeps_obs():
    mudata = _create_mudata(with_activity=True, with_atac=False, fragment_file=None)
    mudata.obs["celltype"] = 1
    mudata.obs["test"] = "test"
    mudata.obs["category"] = ["A" if i < 50 else "B" for i in range(100)]
    mudata.obs["category"] = mudata.obs["category"].astype("category")

    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "celltype" in new_data.obs
    assert "test" in new_data.obs
    assert "category" in new_data.obs
    assert new_data.obs.loc["cell0", "test"] == "test"
    assert new_data.obs.loc["cell0", "celltype"] == 1
    assert pd.api.types.is_categorical_dtype(new_data.obs["category"])


def test_keeps_uns():
    mudata = _create_mudata(with_activity=True, with_atac=False, fragment_file=None)
    mudata.uns["test"] = "test"
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "test" in new_data.uns
    assert new_data.uns["test"] == "test"


def test_keeps_obsm_obsp():
    mudata = _create_mudata(with_activity=True, with_atac=False, fragment_file=None)
    mudata.obsm["test"] = pd.DataFrame(np.ones((mudata.n_obs, 1)), columns=["test"], index=mudata.obs_names)
    mudata.obsp["test"] = np.ones((mudata.n_obs, mudata.n_obs))
    new_data = _safe_mudata(mudata=mudata, modalities=["rna", "activity"])
    assert "test" in new_data.obsm
    assert isinstance(new_data.obsm["test"], pd.DataFrame)
    assert new_data.obsm["test"].shape == (new_data.n_obs, 1)
    assert "test" in new_data.obsm["test"].columns
    assert "test" in new_data.obsp
    assert isinstance(new_data.obsp["test"], np.ndarray)
    assert new_data.obsp["test"].shape == (new_data.n_obs, new_data.n_obs)


def test_a_column_that_cannot_be_aligned_is_dropped_rather_than_raising():
    """Copying `.obs` across is best-effort: a column that will not align is left behind.

    Reached here through a duplicated label on the parent index, which makes the lookup
    return more rows than the new object has. Every other case here aligns cleanly, so the
    guard had never been taken and the copy had only ever been seen succeeding.
    """
    mudata = _create_mudata(with_activity=True, with_atac=False)
    mudata.obs["flag"] = np.arange(len(CELLS))
    mudata.obs.index = pd.Index([CELLS[0]] + CELLS[:-1])

    result = _safe_mudata(mudata, ["rna"])

    assert "flag" not in result.obs.columns
    assert list(result.mod) == ["rna"]
