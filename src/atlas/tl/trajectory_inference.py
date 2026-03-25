from collections.abc import Mapping, Sequence

import numpy as np
import palantir
import pandas as pd
from muon import MuData
from scipy.sparse import csr_matrix, find

from .utils import _assign_state_colors, compute_entropy


class PalantirExtension:
    """Trajectory inference using the Palantir algorithm on ``MuData`` objects.

    This class provides a high-level interface to run Palantir-based trajectory
    inference on multimodal single-cell data stored in a ``MuData`` object.
    It operates directly on the input object and stores all intermediate and
    final results within it.

    Parameters
    ----------
    mudata
        Annotated multimodal data object containing precomputed neighborhood
        graphs and embeddings.

    Notes
    -----
    This implementation is adapted from the original Palantir algorithm
    :cite:`palantir`, with modifications to operate on ``MuData`` objects.
    """

    def __init__(self, mudata: MuData) -> None:
        self._mudata = mudata

    @property
    def mudata(self):
        """Underlying ``MuData`` object.

        Represents multimodal single-cell data
        containing the weighted nearest neighbor graph.

        """
        return self._mudata

    def compute_kernel(
        self,
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


        Returns
        -------
        Updates the original `MuData` object.

            - Anisotropic kernel, scipy.sparse.csr_matrix of shape ``(n_cells, n_cells)`` in `obsp[kernel_key]`.

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

        """
        if distance_key not in self._mudata.obsp.keys():
            raise KeyError(f"{distance_key} not in data.obsp")
        if knn_key not in self._mudata.uns.keys():
            raise KeyError(f"{knn_key} not in data.uns")

        wnn = int(self._mudata.uns[knn_key]["params"]["n_neighbors"])
        if knn is None or knn > wnn:
            knn = wnn

        N = self._mudata.shape[0]
        kNN = self._mudata.obsp[distance_key]
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

        self._mudata.obsp[kernel_key] = kernel

    def compute_diffusion_map(
        self,
        kernel_key: str = "DM_Kernel",
        sim_key: str = "DM_Similarity",
        eigval_key: str = "DM_EigenValues",
        eigvec_key: str = "DM_EigenVectors",
        n_components: int = 10,
        seed: int = 42,
    ) -> None:
        # This function is adapted from Palantir (MIT License).
        # Original source: https://github.com/dpeerlab/Palantir
        # Copyright (c) 2020-present, Dana Pe'er Lab
        # Modifications:
        # - Works on a MuData object
        # - Stores the results from Palantir into the MuData object
        """Compute diffusion maps from a precomputed kernel.

        This function computed the diffusion operator from the kernel stored in `mudata.obsp`.
        Results are stored back into the `MuData` object.

        Parameters
        ----------
        kernel_key
            Key in `mudata.obsp` where the kernel matrix is stored.
        sim_key
            Key in `mudata.obsp` where the diffusion operator is stored.
        eigval_key
            Key in `mudata.uns` where eigenvalues are stored.
        eigvec_key
            Key in `mudata.obsm` where the eigenvectors are stores.
        n_components
            Number of diffusion components to compute.
        seed
            Random seed for reproducibility.


        Returns
        -------
        Updates the original `MuData` object:

            - Sparse similarity matrix of shape ``(n_cells, n_cells)`` in `.obsp[sim_key]`
            - Eigenvectors, pandas.DataFrame of numpy.ndarray of shape  ``(n_cells, n_components)`` in `.obsm[eigval_key]`.
            - Eigenvalues, numpy.ndarray of shape ``(n_components,)`` in `.uns[eigvec_key]`.

        Raises
        ------
        KeyError
            if `kernel_key` is not present in `mudata.obsp`.

        """
        if kernel_key not in self._mudata.obsp.keys():
            raise KeyError(f"{kernel_key} not in data.obsp")

        kernel = self._mudata.obsp[kernel_key]
        res = palantir.utils.diffusion_maps_from_kernel(kernel, n_components, seed)
        self._mudata.obsp[sim_key] = res["T"]
        self._mudata.obsm[eigvec_key] = res["EigenVectors"].set_index(self._mudata.obs.index)
        self._mudata.uns[eigval_key] = res["EigenValues"].values

    def compute_multiscale_space(
        self,
        n_eigs: int | None = None,
        eigval_key: str = "DM_EigenValues",
        eigvec_key: str = "DM_EigenVectors",
        out_key: str = "DM_EigenVectors_multiscaled",
    ) -> None:
        # This function is adapted from Palantir (MIT License).
        # Original source: https://github.com/dpeerlab/Palantir
        # Copyright (c) 2020-present, Dana Pe'er Lab
        # Modifications:
        # - Works on a MuData object
        # - Controls n_eigs against dimensionality of the eigenspace
        # - Calls the original `determine_multiscale_space` from palantir on the eigenvectors dataframe
        # - Stores the results from Palantir into the MuData object
        """Compute the multiscale diffusion distance.

        This function rescales diffusion components using their associated eigenvalues
        to obtain a multiscale representation of the data.

        Parameters
        ----------
        n_eigs
            Number of eigenvalues to use. If `None`, the eigengap heuristic
            is used to obtain a multiscale representation of the data.
        eigval_key
            Key in ``mudata.uns`` where diffusion eigenvalues are stored.
        eigvec_key
            Key in ``mudata.obsm`` where diffusion eigenvectors are stored.
        out_key
            Key in ``mudata.obsm`` where the multiscale representation is stored.


        Returns
        -------
        Updates the original MuData object:
            - Multiscaled distances, pandas.Dataframe of shape ``(n_cells, n_eigs)`` in `obsm[out_key]`.

        Raises
        ------
        KeyError
            If ``eigval_key`` is not present in ``mudata.uns``.
        KeyError
            If ``eigvec_key`` is not present in ``mudata.obsm``.

        """
        if eigval_key not in self._mudata.uns.keys():
            raise KeyError(f"{eigval_key} not in data.uns")
        if eigvec_key not in self._mudata.obsm.keys():
            raise KeyError(f"{eigvec_key} not in data.obsm")

        eigenvectors = (
            pd.DataFrame(self._mudata.obsm[eigvec_key], index=self._mudata.obs_names)
            if not isinstance(self._mudata.obsm[eigvec_key], pd.DataFrame)
            else self._mudata.obsm[eigvec_key]
        )

        if n_eigs is not None and n_eigs > eigenvectors.shape[1]:
            n_eigs = eigenvectors.shape[1]

        dm_dict = {"EigenValues": self._mudata.uns[eigval_key], "EigenVectors": eigenvectors}
        result = palantir.utils.determine_multiscale_space(
            dm_res=dm_dict, n_eigs=n_eigs, eigval_key=eigval_key, eigvec_key=eigvec_key, out_key=out_key
        )
        self._mudata.obsm[out_key] = result

    def run(
        self,
        early_cell: str,
        cluster_key: str | None = None,
        terminal_states: Sequence[str] | Mapping[str, Sequence[str]] | pd.Series | None = None,
        knn: int = 30,
        num_waypoints: int = 1200,
        n_jobs: int = -1,
        scale_components: bool = True,
        use_early_cell_as_start: bool = False,
        max_iterations: int = 25,
        eigvec_multi_key: str = "DM_EigenVectors_multiscaled",
        eigvec_key: str = "DM_EigenVectors",
        pseudotime_key: str = "pseudotime",
        fate_prob_key: str = "fate_probabilities",
        waypoints_key: str = "palantir_waypoints",
        random_state: int = 42,
    ) -> None:
        """Trajectory inference using Palantir (:cite:`palantir`).

        This function:
            - Computes the anisotropic kernel from the weighted nearest neighbor graph connectivites.
            - Derives diffusion components and multiscale space if not available.
            - Runs the original Palantir algorithm.
            - Computed entropy as defined in :cite:`cellrank1`.

        Parameters
        ----------
        early_cell
            Barcode identifier of the initial cell for pseudotime inference.
        cluster_key
            Identifier in `obs` containing information about cells, optional.
        terminal_states
            Terminal states can be defined by the user. When ``None``,
            terminal states are automatically inferred by the algorithm.
        knn:
            Number of nearest neighbors for graph construction in the multiscale space.
        num_waypoints:
            Number of waypoints for trajectory inference. See the original Palantir
            algorithm for an accurate description of the parameter.
        n_jobs:
            Parameter for parallelization.
        scale_components:
            When set to True diffusion components are scaled.
        use_early_cell_as_start:
            If True ther early cell is used as start.
        max_iterations:
            Maximum number of iterations for pseudotime convergence.
        eigvec_multi_key:
            Identifier of the multiscale space matrix in ``.obsm``.
        pseudotime_key:
            Key in ``.obs`` where pseudotime results are stored.
        fate_prob_key:
            Key in ``.obsm`` where fate probabilities are stored.
        waypoints_key:
            Key in ``.uns`` where waypoints are stored if provided.
        random_state:
            Seed for reproducibility.

        Returns
        -------
        Updates the `MuData` object:
            - Pseudotime is added in `.obs[pseudotime_key]`.
            - Fate probabilites, a pandas.Dataframe of shape ``(n_cells, n_terminal_states)`` in `obsm[fate_prob_key]`.
            - List of inferred waypoints in `.uns[waypoints_key]`.
            - Shannon entropy and KL-divergence values, respectively, in `.obs["shannon_entropy"]` and `.obs["kl_divergence"]`.
            - Initial and terminal cell identifiers, respectively, in `.uns["initial_states"]` and `uns["terminal_states"]`.

        Notes
        -----
        If the multiscale space identified by ``eigvec_multi_key`` is not present,
        the anisotropc kernel, the diffusion components and the multiscaled distances are inferred
        using the default parameters on the graph stored as "wnn_distances" in ``.obsp``.

        """
        # If multiscale distances are not present they are automatically computed
        # using standard parameters
        if eigvec_multi_key not in self._mudata.obsm:
            self.compute_kernel()
            self.compute_diffusion_map(seed=random_state)
            self.compute_multiscale_space(out_key=eigvec_multi_key)

        res = palantir.core.run_palantir(
            data=self._mudata.obsm[eigvec_multi_key],
            early_cell=early_cell,
            terminal_states=terminal_states,
            knn=knn,
            num_waypoints=num_waypoints,
            n_jobs=n_jobs,
            scale_components=scale_components,
            use_early_cell_as_start=use_early_cell_as_start,
            max_iterations=max_iterations,
            eigvec_key=eigvec_key,
            pseudo_time_key=pseudotime_key,
            entropy_key="palantir_entropy",
            fate_prob_key=fate_prob_key,
            save_as_df=True,
            waypoints_key=None,
            seed=random_state,
        )

        self._mudata.obs[pseudotime_key] = res.pseudotime
        self._mudata.uns[waypoints_key] = res.waypoints.values
        if isinstance(terminal_states, pd.Series):
            res.branch_probs.columns = terminal_states[res.branch_probs.columns]

        # Renames inferred states according to the cell type in .obs["cluster_key"]
        # rather by the standard default behavior of Palantir, i.e., using the barcode
        if cluster_key is not None and cluster_key in self._mudata.obs.columns:
            cell_to_cluster = self._mudata.obs[cluster_key]
            terminal_states = {}
            for cell in res.branch_probs.columns:
                cluster = cell_to_cluster.loc[cell]
                terminal_states.setdefault(cluster, []).append(cell)

            initial_states = {cell_to_cluster.loc[early_cell]: [early_cell]}
            res.branch_probs.columns = cell_to_cluster.loc[res.branch_probs.columns].values
            fate_probs = res.branch_probs.groupby(level=0, axis=1).sum()

        else:
            terminal_states = {cell: [cell] for cell in res.branch_probs}
            initial_states = {early_cell: [early_cell]}
            fate_probs = res.branch_probs

        self._mudata.uns["initial_states"] = initial_states
        self._mudata.uns["terminal_states"] = terminal_states
        self._mudata.obsm[fate_prob_key] = fate_probs
        _assign_state_colors(self._mudata)

        compute_entropy(mudata=self._mudata, fate_probability_key=fate_prob_key)
