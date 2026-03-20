import numpy as np
from muon import MuData
from scipy.sparse import csr_matrix, find


def compute_kernel(
    mudata: MuData,
    knn_key: str = "wnn",
    distance_key: str = "wnn_distances",
    knn: int | None = None,
    alpha: float = 0,
    kernel_key: str = "DM_Kernel",
) -> None:
    # This function is adapted from Palantir (MIT License).
    # Original source: https://github.com/dpeerlab/Palantir
    # Copyright (c) 2020-present, Dana Pe'er Lab
    # Modifications:
    # - Works on a MuData object
    # - Does not compute neighborhood graph and
    #       it accounts for the graph to be already present
    # - Avoids sklearn-based backend
    """Compute an adaptive Gaussian kernel from a kNN graph.

    This function builds a symmetric affinity matrix using an adaptive
    bandwidth scheme, following the approach described in :cite:`palantir`.
    The kernel is constructed from a k-nearest neighbors distance graph
    stored in ``mudata.obsp``.

    Parameters
    ----------
    mudata
        MuData object containing the neighborhood graph.
    knn_key
        Key in ``mudata.uns`` where neighborhood parameters are stored.
        Must contain ``['params']['n_neighbors']``. Default is ``"wnn"``.
    distance_key
        Key in ``mudata.obsp`` where the kNN distance matrix is stored.
        Default is ``"wnn_distances"``.
    knn
        Number of nearest neighbors used to define the adaptive bandwidth.
        If ``None`` or larger than the number of neighbors stored in
        ``mudata.uns[knn_key]['params']['n_neighbors']``, it is set to that value.
    alpha
        Normalization parameter for the diffusion operator. If ``alpha > 0``,
        applies degree normalization to the kernel. Default is ``0``.
    kernel_key
        Key in ``mudata.obsp`` where the resulting kernel matrix is stored.
        Default is ``"DM_Kernel"``.

    Raises
    ------
    KeyError
        If ``distance_key`` is not present in ``mudata.obsp`` or
        ``knn_key`` is not present in ``mudata.uns``.

    Notes
    -----
    The kernel is computed using an adaptive bandwidth defined as the
    distance to the ``floor(knn / 3)``-th nearest neighbor for each cell.
    The resulting affinity matrix is symmetrized and optionally normalized.

    Adds
    ----
    mudata.obsp[kernel_key]
        Sparse symmetric kernel matrix of shape ``(n_cells, n_cells)``.
    """
    if distance_key not in mudata.obsp.keys():
        raise KeyError(f"{distance_key} not in data.obsp")
    if knn_key not in mudata.uns.keys():
        raise KeyError(f"{knn_key} not in data.uns")

    wnn = int(mudata.uns[knn_key]["params"]["n_neighbors"])
    if knn is None or knn > wnn:
        knn = wnn

    N = mudata.shape[0]
    kNN = mudata.obsp[distance_key]
    adaptive_k = int(np.floor(knn / 3))
    adaptive_std = np.zeros(N)
    for i in np.arange(N):
        adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[adaptive_k - 1]
    x, y, dists = find(kNN)
    dists /= adaptive_std[x]
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])
    kernel = W + W.T
    if alpha > 0:
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)

    mudata.obsp[kernel_key] = kernel
