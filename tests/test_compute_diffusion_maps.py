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

# Eigenvalues produced by `compute_diffusion_maps` for the kernel built in
# `test_eigenvalues_match_reference`, with SEED and three components. Recorded against
# Palantir 1.4.5, the floor this package declares.
REFERENCE_EIGENVALUES = np.array([1.0, 0.064443670904, 0.062733464624])

# Repeated runs in one process reproduce the reference exactly, so the tolerance exists
# only to absorb variation in ARPACK convergence across platforms and BLAS builds.
# Note what this does NOT catch: substituting Palantir's pre-1.4.5 global-seed RNG for
# the local generator introduced in 1.4.5 moves these eigenvalues by ~1e-7 on
# well-conditioned inputs like this one, which is below EIGENVALUE_ATOL. Detecting that
# boundary needs a tolerance near 1e-8, which platform noise would make flaky. This
# assertion guards against gross numerical drift; the version floor in pyproject.toml is
# what excludes the differing RNG.
EIGENVALUE_ATOL = 1e-6


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


def test_eigenvalues_match_reference():
    kernel_key = "KERNEL"
    eigval_key = "EIGENVALUES"
    n_components = 3
    rng = np.random.default_rng(SEED)
    mdata = _create_mudata()
    mdata.obsp[kernel_key] = csr_matrix(rng.uniform(0, 1, size=(len(CELLS), len(CELLS))))

    pext = PalantirExtension(mudata=mdata)
    pext.compute_diffusion_maps(kernel_key=kernel_key, eigval_key=eigval_key, n_components=n_components, seed=SEED)

    np.testing.assert_allclose(mdata.uns[eigval_key], REFERENCE_EIGENVALUES, atol=EIGENVALUE_ATOL)


def test_eigenvalues_are_reproducible_across_calls():
    kernel_key = "KERNEL"
    eigval_key = "EIGENVALUES"
    n_components = 3
    rng = np.random.default_rng(SEED)
    kernel = csr_matrix(rng.uniform(0, 1, size=(len(CELLS), len(CELLS))))

    results = []
    for _ in range(2):
        mdata = _create_mudata()
        mdata.obsp[kernel_key] = kernel
        pext = PalantirExtension(mudata=mdata)
        pext.compute_diffusion_maps(kernel_key=kernel_key, eigval_key=eigval_key, n_components=n_components, seed=SEED)
        results.append(mdata.uns[eigval_key])

    np.testing.assert_array_equal(results[0], results[1])
