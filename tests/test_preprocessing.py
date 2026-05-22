from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

import atlas
from atlas.pp import preprocessing

CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
PEAKS = [f"peak{i}" for i in range(50)]

FEATURES = pd.DataFrame(
    {
        "chrom": ["chr1"] * len(GENES),
        "start": np.arange(len(GENES)),
        "end": np.arange(len(GENES)) + 1,
    }
)

SEED, n_comps = 42, 10
n_pcs_rna, n_pcs_act = 5, 5
knn_rna, knn_act, n_neighbors = 5, 5, 5
n_multineighbors, n_bandwidth_neighbors = 30, 5


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


def test_package_has_version():
    assert atlas.__version__ is not None


def test_requires_rna():
    mdata = MuData({"dummy": AnnData(np.zeros((1, 1)))})
    with pytest.raises(KeyError, match="rna"):
        preprocessing(mdata)


def test_requires_atac():
    mdata = _create_mudata(with_activity=False, with_atac=False)
    with pytest.raises(KeyError, match="atac"):
        preprocessing(mdata)


def test_requires_fragments():
    mdata = _create_mudata(with_activity=False, with_atac=True)

    with pytest.raises(ValueError, match="fragment"):
        preprocessing(mdata)


def test_requires_features():
    mdata = _create_mudata(with_activity=False, with_atac=True)
    with pytest.raises(ValueError, match="Feature"):
        preprocessing(mdata, fragment_path="tests/fragments.tsv.gz", features=None)


def test_copy_true():
    mdata = _create_mudata(with_activity=True, with_atac=True)
    sc.pp.normalize_total(mdata["rna"])
    sc.pp.pca(mdata["rna"])
    result = preprocessing(
        mdata,
        features=FEATURES,
        knn_rna=knn_rna,
        knn_act=knn_act,
        n_pcs_rna=n_pcs_rna,
        n_pcs_act=n_pcs_act,
        n_neighbors=n_neighbors,
        n_multineighbors=n_multineighbors,
        n_bandwidth_neighbors=n_bandwidth_neighbors,
        copy=True,
    )
    assert "atac" not in result.mod
    assert "atac" in mdata.mod


def test_copy_false():
    mdata = _create_mudata(with_activity=True, with_atac=True)
    sc.pp.normalize_total(mdata["rna"])
    sc.pp.pca(mdata["rna"], n_comps=n_comps, random_state=SEED)
    sc.pp.normalize_total(mdata["activity"])
    sc.pp.pca(mdata["activity"], n_comps=n_comps, random_state=SEED)
    result = preprocessing(
        mdata,
        features=FEATURES,
        knn_rna=knn_rna,
        knn_act=knn_act,
        n_pcs_rna=n_pcs_rna,
        n_pcs_act=n_pcs_act,
        n_neighbors=n_neighbors,
        n_multineighbors=n_multineighbors,
        n_bandwidth_neighbors=n_bandwidth_neighbors,
        copy=False,
    )
    assert "atac" not in result.mod
    assert "atac" in mdata.mod


@patch("atlas.pp.basic.mu.atac.tl.count_fragments_features")
def test_activity(mock_func):
    mock_func.return_value = _generate_activity()
    mdata = _create_mudata(with_atac=True, fragment_file="tests/fragments.tsv.gz")
    sc.pp.normalize_total(mdata["rna"])
    sc.pp.pca(mdata["rna"], n_comps=n_comps, random_state=SEED)
    result = preprocessing(
        mdata,
        features=FEATURES,
        knn_rna=knn_rna,
        knn_act=knn_act,
        n_pcs_rna=n_pcs_rna,
        n_pcs_act=n_pcs_act,
        n_neighbors=n_neighbors,
        n_multineighbors=n_multineighbors,
        n_bandwidth_neighbors=n_bandwidth_neighbors,
        copy=False,
    )
    assert "activity" in result.mod
    assert "rna" in result.mod
    assert "atac" not in result.mod
    assert "X_umap" in result.obsm
    assert "wnn" in result.uns
    assert "wnn_connectivities" in result.obsp
    assert "wnn_distances" in result.obsp
