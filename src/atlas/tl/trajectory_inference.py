from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import palantir
import pandas as pd
from anndata import AnnData
from cellrank.estimators import GPCCA
from cellrank.kernels import PseudotimeKernel
from muon import MuData
from scipy.sparse import csr_matrix, find

from .utils import _assign_state_colors, _invert_assignment, compute_entropy


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
            - Eigenvectors, :class:`pandas.DataFrame` or :class:`numpy.ndarray` of shape  ``(n_cells, n_components)`` in `.obsm[eigval_key]`.
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

            - Multiscaled distances, :class:`pandas.DataFrame` of shape ``(n_cells, n_eigs)`` in `obsm[out_key]`.

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
            - Fate probabilites, a :class:`pandas.DataFrame` of shape ``(n_cells, n_terminal_states)`` in `obsm[fate_prob_key]`.
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
            fate_probs = res.branch_probs.T.groupby(level=0).sum().T

        else:
            terminal_states = {cell: [cell] for cell in res.branch_probs}
            initial_states = {early_cell: [early_cell]}
            fate_probs = res.branch_probs

        self._mudata.uns["initial_states"] = initial_states
        self._mudata.uns["terminal_states"] = terminal_states
        self._mudata.obsm[fate_prob_key] = fate_probs
        _assign_state_colors(self._mudata)

        compute_entropy(mudata=self._mudata, fate_probability_key=fate_prob_key)


class CellRankExtension:
    """Trajectory inference using the CellRank Pseudotime Kernel on ``MuData`` objects.

    This class provides a high-level interface to run Psseudotime-based trajectory
    inference on multimodal single-cell data stored in a ``MuData``.
    It operates directly on the input object and stores all intermediate and
    final results within it.

    Parameters
    ----------
    mudata
        Annotated multimodal data object containing precomputed neighborhood
        graphs and embeddings.

    Notes
    -----
    This implementation is adapted from the original CellRank algorithm
    :cite:`cellrank2`, with modifications to operate on ``MuData`` objects.
    """

    def __init__(self, mudata: MuData):
        self._mudata = mudata

    @property
    def mudata(self):
        """Underlying ``MuData`` object.

        Represents multimodal single-cell data
        containing the weighted nearest neighbor graph.

        """
        return self._mudata

    @property
    def kernel(self):
        """Underlying ``PseudotimeKernel`` object."""
        return self._kernel

    @property
    def time_key(self):
        """Key in ``mudata.obs`` storing cell-wise time."""
        return self._time_key

    @property
    def cluster_key(self):
        """Key in ``mudata.obs`` storing cell-wise categories."""
        return self._cluster_key

    @property
    def fate_key(self):
        """Key in ``mudata.obsm`` storing cell-fate probabilities."""
        return self._fate_key

    def compute_kernel(
        self,
        connectivity_key: str = "wnn_connectivities",
        time_key: str = "pseudotime",
        cluster_key: str | None = None,
        backward: bool = False,
        threshold_scheme: Literal["soft", "hard"] = "hard",
        frac_to_keep: float = 0.3,
        b: float = 10.0,
        nu: float = 0.5,
        n_jobs: int = -1,
    ) -> None:
        # This function is adapted from CellRab (BSD 3-Clause License).
        # Original source: https://github.com/scverse/cellrank
        # Copyright (c) 2019, Theis Lab
        # Modifications:
        # - Works on a MuData object
        # - Instances the original PseudotimeKernel
        # - Calls the original `PseudotimeKernel.compute_kernel` function
        """Compute a pseudotime-based transition kernel.

        This function constructs a :class:`cellrank.kernels.PseudotimeKernel`
        using a precomputed connectivity graph and pseudotime stored in the
        underlying ``MuData`` object. The kernel is used to model directed
        transitions between cells based on their pseudotemporal ordering.

        Parameters
        ----------
        connectivity_key
            Key in ``mudata.obsp`` where the connectivity matrix corresponding to the
            WNN is stored.
        time_key
            Key in ``mudata.obs`` where pseudotime values are stored.
        cluster_key
            Optional key in ``mudata.obs`` containing cell annotations.
            If provided, clusters are treated as categorical groups.
        backward
            Whether to compute the backward process instead of the forward
            pseudotime dynamics. Refer to :cite:`cellrank2` for more information.
        threshold_scheme
            Scheme used to sparsify the transition matrix. One of
            ``"soft"`` or ``"hard"``.
        frac_to_keep
            Fraction of transitions to retain when using thresholding.
        b
            Parameter controlling the steepness of the logistic function
            used in soft thresholding.
        nu
            Parameter controlling the width of the kernel.
        n_jobs
            Number of parallel jobs used for computation.

        Returns
        -------
        Updates the `MuData` object:

            - Stores a :class:`cellrank.kernels.PseudotimeKernel` instance
              in ``self._kernel`` containing the computed transition matrix.

        Raises
        ------
        KeyError
            If ``connectivity_key`` is not present in ``mudata.obsp``.
        KeyError
            If ``cluster_key`` is provided but not present in ``mudata.obs``.

        Notes
        -----
        This method creates a temporary :class:`~anndata.AnnData` object to
        interface with CellRank, while preserving the original ``MuData``
        structure.


        """
        if connectivity_key not in self._mudata.obsp.keys():
            raise KeyError(f"{connectivity_key} not in mudata.obsp")
        if time_key not in self._mudata.obs.columns:
            raise KeyError(f"{connectivity_key} not in mudata.obs")
        if cluster_key is not None and cluster_key not in self.mudata.obs.columns:
            raise KeyError(f"{cluster_key} not in mudata.obs")

        self._time_key = time_key
        self._cluster_key = cluster_key

        _tmp = AnnData(
            X=csr_matrix((self.mudata.n_obs, self.mudata["rna"].n_vars)),
            obs=self.mudata.obs.copy(),
            var=pd.DataFrame([], index=self.mudata["rna"].var_names),
        )
        _tmp.obsp[connectivity_key] = self.mudata.obsp[connectivity_key]

        if cluster_key is not None:
            _tmp.obs[cluster_key] = _tmp.obs[cluster_key].astype("category")

        self._kernel = PseudotimeKernel(adata=_tmp, time_key=time_key, conn_key=connectivity_key, backward=backward)
        self._kernel.compute_transition_matrix(
            threshold_scheme=threshold_scheme, frac_to_keep=frac_to_keep, b=b, nu=nu, n_jobs=n_jobs
        )

    def run(
        self,
        n_components: int = 20,
        initial_states: dict[str, Sequence[str]] | None = None,
        terminal_states: dict[str, Sequence[str]] | None = None,
        initial_distribution: np.ndarray | None = None,
        method: Literal["krylov", "brandts"] = "krylov",
        sorting_strategy: Literal["LM", "LR"] = "LR",
        eigengap_weight: float = 1.0,
        verbose: bool | None = None,
        n_cells: int = 30,
        n_states: int | Sequence[int] | None = None,
        allow_overlap: bool = False,
        alpha: float = 1.0,
        stability_threshold: float = 0.96,
        n_terminal_states: int | None = None,
        n_initial_states: int | None = None,
        solver: Literal["direct", "gmres", "lgmres", "bicgstab", "gcrotmk"] = "gmres",
        use_petsc: bool = True,
        n_jobs: int = -1,
        tol: float = 1e-6,
        preconditioner: str | None = None,
    ) -> None:
        """Run trajectory inference using GPCCA on a precomputed kernel.

        This method applies the Generalized Perron Cluster Cluster Analysis (GPCCA)
        algorithm to identify macrostates, infer initial and terminal states, and
        compute fate probabilities. Results are stored directly in the underlying
        ``MuData`` object.

        Depending on the provided inputs, the method either:
            - uses user-defined initial and terminal states, or
            - automatically infers macrostates and predicts initial and terminal states.

        Parameters
        ----------
        n_components
            Number of Schur vectors to compute for the coarse-graining step.
        initial_states
            Mapping of initial states to cell identifiers. If ``None``, initial states
            are inferred automatically.
        terminal_states
            Mapping of terminal states to cell identifiers. If ``None``, terminal states
            are inferred automatically.
        initial_distribution
            Optional initial distribution over cells used for the Schur decomposition.
        method
            Method used to compute the Schur decomposition.
        sorting_strategy
            Strategy to sort eigenvalues (e.g. largest magnitude ``"LM"`` or largest real part ``"LR"``).
        eigengap_weight
            Weight controlling the eigengap heuristic used during coarse-graining. Refer to
            :meth:`cellrank.estimators.GPCCA.compute_schur` for more information.
        verbose
            Whether to print progress information.
        n_cells
            Number of cells to sample when identifying representative states.
        n_states
            Number of macrostates to compute. If ``None``, it is determined automatically according to
            minChi and crispness, refer to :cite:`cellrank1` and :meth:`cellrank.estimators.GPCCA.compute_macrostates`
            for more information.
        allow_overlap
           Whether cells can belong to multiple states.
        alpha
            Parameter controlling terminal state prediction. Refer to :meth:`cellrank.estimators.GPCCA.predict_terminal_states`
            and :meth:`cellrank.estimators.GPCCA.predict_initial_states`.
        stability_threshold
            Threshold used to identify stable terminal states.
        n_terminal_states
            Number of terminal states to predict when not provided.
        n_initial_states
            Number of initial states to predict when not provided.
        solver
            Linear solver used for fate probability computation.
        use_petsc
            Whether to use :mod:`petsc4py` or :mod:`scipy` for solving linear systems.
        n_jobs
            Number of parallel jobs for the iterative solver.
        tol
            Convergence tolerance for the iterative solver.
        preconditioner
            Optional preconditioner for the solver. See :meth:`cellrank.estimators.GPCCA.compute_fate_probabilities`.

        Returns
        -------
        Updates the original ``MuData`` object:

            - Fate probabilities stored as a :class:`pandas.DataFrame` in ``.obsm["fate_probabilities"]``.
            - Initial states stored in ``.uns["initial_states"]``.
            - Terminal states stored in ``.uns["terminal_states"]``.
            - Intermediate states (if inferred) stored in ``.uns["intermediate_states"]``.
            - State-specific colors stored in ``.uns["fate_state_colors"]``.
            - Entropy measures (e.g. Shannon entropy, KL divergence) stored in ``.obs``.

        Raises
        ------
        AttributeError
            If the kernel has not been computed prior to calling this method.

        Notes
        -----
        This implementation relies on :class:`cellrank.estimators.GPCCA` to perform
        coarse-graining of the Markov chain and infer lineage relationships.

        If both ``initial_states`` and ``terminal_states`` are provided, no automatic
        state inference is performed. Otherwise, macrostates and lineage-driving states
        are inferred from the data.
        """
        if not hasattr(self, "kernel"):
            raise AttributeError("Kernel not found. Run compute_kernel first.")

        _G = GPCCA(self.kernel)
        _G.compute_schur(
            n_components,
            initial_distribution=initial_distribution,
            method=method,
            which=sorting_strategy,
            alpha=eigengap_weight,
            verbose=verbose,
        )

        if terminal_states is not None and initial_states is not None:
            _G.set_initial_states(
                states=initial_states, n_cells=n_cells, allow_overlap=allow_overlap, cluster_key=self.cluster_key
            )
            _G.set_terminal_states(
                states=terminal_states, n_cells=n_cells, allow_overlap=allow_overlap, cluster_key=self.cluster_key
            )
        else:
            _G.compute_macrostates(n_states=n_states, cluster_key=self.cluster_key)
            _G.predict_terminal_states(
                n_cells=n_cells,
                alpha=alpha,
                stability_threshold=stability_threshold,
                n_states=n_terminal_states,
                allow_overlap=allow_overlap,
            )
            _G.predict_initial_states(n_states=n_initial_states, n_cells=n_cells, allow_overlap=allow_overlap)

        _G.compute_fate_probabilities(
            solver=solver, use_petsc=use_petsc, n_jobs=n_jobs, tol=tol, preconditioner=preconditioner
        )

        self.mudata.obsm["fate_probabilities"] = pd.DataFrame(
            _G.fate_probabilities.X, index=self.mudata.obs_names, columns=_G.fate_probabilities.names
        )

        self.mudata.uns["initial_states"] = _invert_assignment(_G.initial_states)
        self.mudata.uns["terminal_states"] = _invert_assignment(_G.terminal_states)

        if terminal_states is None and initial_states is None:
            intermediate = _G.macrostates[(_G.initial_states.isna()) & (_G.terminal_states.isna())]
            intermediate = intermediate.cat.remove_unused_categories()
            self.mudata.uns["intermediate_states"] = _invert_assignment(intermediate)
        else:
            self.mudata.uns["intermediate_states"] = {}

        self._fate_key = "fate_probabilities"
        _assign_state_colors(self.mudata)

        self.compute_entropy(fate_prob_key="fate_probabilities")
