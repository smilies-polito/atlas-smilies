import numpy as np
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import PalantirExtension

SEED, K = 42, 10
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]

EIGENVALUES = np.array([1, 0.95, 0.5])
EIGENVECTORS = np.random.default_rng(SEED).random((len(CELLS), 3))


def _create_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    return mudata


def test_missing_eigenvalues():
    pext = PalantirExtension(_create_mudata())
    with pytest.raises(KeyError, match="uns"):
        pext.compute_multiscale_space()


def test_missing_eigenvectors():
    eigval_key = "eigenvalues"
    mdata = _create_mudata()
    mdata.uns[eigval_key] = EIGENVALUES
    pext = PalantirExtension(mdata)

    with pytest.raises(KeyError, match="obsm"):
        pext.compute_multiscale_space(eigval_key=eigval_key)


def test_ok():
    eigval_key, eigvec_key = "eigenvalues", "eigenvectors"
    out_key, n_eigs = "output", 2
    mdata = _create_mudata()
    mdata.uns[eigval_key] = EIGENVALUES
    mdata.obsm[eigvec_key] = EIGENVECTORS
    pext = PalantirExtension(mdata)
    pext.compute_multiscale_space(out_key=out_key, n_eigs=n_eigs, eigval_key=eigval_key, eigvec_key=eigvec_key)
    assert out_key in mdata.obsm


def test_exceeding_components():
    eigval_key, eigvec_key = "eigenvalues", "eigenvectors"
    out_key, n_eigs = "output", 10
    mdata = _create_mudata()
    mdata.uns[eigval_key] = EIGENVALUES
    mdata.obsm[eigvec_key] = EIGENVECTORS
    pext = PalantirExtension(mdata)
    pext.compute_multiscale_space(out_key=out_key, n_eigs=n_eigs, eigval_key=eigval_key, eigvec_key=eigvec_key)
    assert out_key in mdata.obsm
