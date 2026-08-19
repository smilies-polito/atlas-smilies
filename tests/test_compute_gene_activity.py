from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData

from atlas.pp import compute_gene_activity, preprocessing

SEED = 42
CELLS = [f"cell{i}" for i in range(60)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
PEAKS = [f"peak{i}" for i in range(30)]
FRAGMENTS = "tests/fragments.tsv.gz"

FEATURES = pd.DataFrame(
    {
        "chrom": ["chr1"] * len(GENES),
        "start": np.arange(len(GENES)),
        "end": np.arange(len(GENES)) + 1,
    }
)


def _generate_activity() -> AnnData:
    activity = AnnData(np.random.default_rng(SEED).random((len(CELLS), len(ACTIVITY_VAR))))
    activity.obs_names, activity.var_names = CELLS, ACTIVITY_VAR
    return activity


def _create_mudata(with_atac: bool = True, atac_key: str = "atac") -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    rna.obs_names, rna.var_names = CELLS, GENES
    mods = {"rna": rna}

    if with_atac:
        atac = AnnData(rng.random((len(CELLS), len(PEAKS))))
        atac.obs_names, atac.var_names = CELLS, PEAKS
        mods[atac_key] = atac
    return MuData(mods)


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_activity_is_derived(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(_create_mudata(), features=FEATURES, fragment_path=FRAGMENTS)
    assert "activity" in result.mod
    assert result["activity"].shape == (len(CELLS), len(ACTIVITY_VAR))
    assert "X_pca" in result["activity"].obsm


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_no_graph_is_built(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(_create_mudata(), features=FEATURES, fragment_path=FRAGMENTS)
    assert "wnn_distances" not in result.obsp
    assert "wnn" not in result.uns


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
@patch("atlas.pp.basic.mu.atac.tl.count_fragments_features")
def test_matches_the_modality_preprocessing_derives(mock_basic, mock_new):
    mock_basic.return_value = _generate_activity()
    mock_new.return_value = _generate_activity()

    old = preprocessing(
        _create_mudata(),
        features=FEATURES,
        fragment_path=FRAGMENTS,
        knn_rna=5,
        knn_act=5,
        n_pcs_rna=5,
        n_pcs_act=5,
        random_state=SEED,
    )
    new = compute_gene_activity(_create_mudata(), features=FEATURES, fragment_path=FRAGMENTS, random_state=SEED)

    assert np.allclose(old["activity"].X, new["activity"].X)
    assert np.allclose(old["activity"].obsm["X_pca"], new["activity"].obsm["X_pca"])


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_n_comps_sets_the_reduction_width(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(
        _create_mudata(), features=FEATURES, fragment_path=FRAGMENTS, n_comps=4, random_state=SEED
    )
    assert result["activity"].obsm["X_pca"].shape[1] == 4


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_n_comps_defaults_to_the_scanpy_choice(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(_create_mudata(), features=FEATURES, fragment_path=FRAGMENTS, random_state=SEED)

    reference = _generate_activity()
    sc.pp.normalize_total(reference)
    sc.pp.pca(reference, random_state=SEED)

    assert result["activity"].obsm["X_pca"].shape == reference.obsm["X_pca"].shape
    assert np.allclose(result["activity"].obsm["X_pca"], reference.obsm["X_pca"])


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_out_key(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(
        _create_mudata(), features=FEATURES, fragment_path=FRAGMENTS, out_key="ga", random_state=SEED
    )
    assert "ga" in result.mod
    assert "activity" not in result.mod


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_atac_key(mock_counts):
    mock_counts.return_value = _generate_activity()
    result = compute_gene_activity(
        _create_mudata(atac_key="peaks"),
        features=FEATURES,
        fragment_path=FRAGMENTS,
        atac_key="peaks",
        random_state=SEED,
    )
    assert "activity" in result.mod


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_copy_true(mock_counts):
    mock_counts.return_value = _generate_activity()
    mdata = _create_mudata()
    result = compute_gene_activity(mdata, features=FEATURES, fragment_path=FRAGMENTS, copy=True)
    assert "activity" in result.mod
    assert "activity" not in mdata.mod


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_copy_false(mock_counts):
    mock_counts.return_value = _generate_activity()
    mdata = _create_mudata()
    result = compute_gene_activity(mdata, features=FEATURES, fragment_path=FRAGMENTS, copy=False)
    assert result is mdata
    assert "activity" in mdata.mod


def test_requires_atac():
    with pytest.raises(KeyError, match="atac"):
        compute_gene_activity(_create_mudata(with_atac=False), features=FEATURES)


def test_requires_fragments():
    with pytest.raises(ValueError, match="Fragment file"):
        compute_gene_activity(_create_mudata(), features=FEATURES, fragment_path=None)


def test_requires_features():
    mdata = _create_mudata()
    with pytest.raises(ValueError, match="Feature dataframe"):
        compute_gene_activity(mdata, features=None, fragment_path=FRAGMENTS)
    assert "activity" not in mdata.mod


@patch("atlas.pp._preprocessing.mu.atac.tl.locate_file")
@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_a_fragment_file_already_recorded_is_not_located_again(mock_counts, mock_locate):
    """A fragment file already in ``.uns['files']`` is used as it stands.

    Every other test here arrives without one, so the guard was only ever seen failing
    and this path had never run.
    """
    mock_counts.return_value = _generate_activity()
    mudata = _create_mudata()
    mudata["atac"].uns["files"] = {"fragments": FRAGMENTS}

    result = compute_gene_activity(mudata, features=FEATURES)

    assert mock_locate.call_count == 0
    assert "activity" in result.mod


@patch("atlas.pp._preprocessing.mu.atac.tl.count_fragments_features")
def test_a_recorded_fragment_file_makes_the_path_argument_unnecessary(mock_counts):
    """``fragment_path`` is only required when the object does not already carry one."""
    mock_counts.return_value = _generate_activity()
    mudata = _create_mudata()
    mudata["atac"].uns["files"] = {"fragments": FRAGMENTS}

    assert compute_gene_activity(mudata, features=FEATURES, fragment_path=None) is not None
