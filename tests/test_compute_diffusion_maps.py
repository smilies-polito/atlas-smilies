import numpy as np
import pytest
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import PalantirExtension

SEED, K = 42, 10
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]


def _create_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    return mudata


def test_missing_kernel():
    mdata = _create_mudata()
    pext = PalantirExtension(mudata=mdata)

    with pytest.raises(KeyError, match="data.obsp"):
        pext.compute_diffusion_maps()


def test_ok():
    kernel_key, sim_key = (
        "KERNEL",
        "SIMILARITY",
    )
    eigval_key = "EIGENVALUES"
    eigvec_key = "EIGENVECTORS"
    rng = np.random.default_rng(SEED)
    n_components = 2
    mdata = _create_mudata()
    mdata.obsp[kernel_key] = csr_matrix(rng.uniform(0, 1, size=(len(CELLS), len(CELLS))))

    pext = PalantirExtension(mudata=mdata)

    pext.compute_diffusion_maps(
        kernel_key=kernel_key,
        sim_key=sim_key,
        eigval_key=eigval_key,
        eigvec_key=eigvec_key,
        n_components=n_components,
        seed=SEED,
    )

    assert sim_key in mdata.obsp
    assert mdata.obsp[sim_key].shape == (len(CELLS), len(CELLS))
    assert eigvec_key in mdata.obsm
    assert mdata.obsm[eigvec_key].shape == (len(CELLS), n_components)
    assert eigval_key in mdata.uns
    assert len(mdata.uns[eigval_key]) == n_components
