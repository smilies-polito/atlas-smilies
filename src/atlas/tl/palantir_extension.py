import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import palantir
import pandas as pd
from muon import MuData
from scipy.sparse import csr_matrix, find

from .utils import (
    _assign_state_colors,
    _deprecated_key_arg,
    _disambiguate_names,
    _palantir_anndata,
    _resolve_graph_key,
    _resolve_overlap,
    _states_to_column,
    compute_entropy,
)

#: Key under which :func:`palantir.utils.fallback_terminal_cell` looks up the multiscale
#: space. Palantir below 1.4.5 does not forward the ``eigvec_key`` of
#: :func:`palantir.utils.early_cell` to that fallback, so the representation is
#: registered under this name as well. Harmless on 1.4.5 and above, where it is fixed.
_PALANTIR_FALLBACK_EIGVEC_KEY = "DM_EigenVectors_multiscaled"


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
        key: str | None = None,
        knn_key: str | None = None,
        distance_key: str | None = None,
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
        """Compute an adaptive Gaussian kernel from a neighbors graph.

        This function builds a symmetric affinity matrix using an adaptive
        bandwidth scheme, following the approach described in :cite:`palantir`.
        The kernel is constructed from a nearest neighbors distance graph
        stored in ``mudata.obsp``.

        .. deprecated:: 1.1.0
            The `distance_key` and `knn_key` parameters are deprecated and
            will be removed in version 2.0.0. They are retained in version
            1.1.0 for backwards compatibility. Use `key` to adopt the new behavior.

        Parameters
        ----------
        key
            Key in ``mudata.uns`` where the graph is recorded. The distance matrix and the
            neighborhood size are resolved from it, so nothing else is needed to identify
            the graph.
        knn_key
             Key in ``mudata.uns`` where neighborhood parameters are stored.
             Must contain ``['params']['n_neighbors']``.
             Deprecated in 1.1.0 and will be removed in version 2.0.0.
        distance_key
             Key in ``mudata.obsp`` where the kNN distance matrix is stored.
             Deprecated in 1.1.0 and will be removed in version 2.0.0.
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
        _deprecated_key_arg("knn_key", knn_key, "key", key)
        _deprecated_key_arg("distance_key", distance_key, "key", key)

        record_key = knn_key if knn_key is not None else (key if key is not None else "wnn")
        if record_key not in self._mudata.uns.keys():
            raise KeyError(f"{record_key} not in data.uns")

        if distance_key is None:
            distance_key = _resolve_graph_key(self._mudata, record_key, "distances")
        elif distance_key not in self._mudata.obsp.keys():
            raise KeyError(f"{distance_key} not in data.obsp")

        wnn = int(self._mudata.uns[record_key]["params"]["n_neighbors"])
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

    def compute_diffusion_maps(
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

    def early_cell(
        self,
        celltype: str,
        cluster_key: str = "celltype",
        eigvec_multi_key: str = "DM_EigenVectors_multiscaled",
        fallback_seed: int | None = None,
        random_state: int = 42,
    ) -> str:
        """Select the initial cell for :meth:`~atlas.tl.PalantirExtension.run` from the multiscale diffusion space.

        Identifies the cell of ``celltype`` lying at an extreme of the state space
        represented by the diffusion components, which is the cell
        :meth:`atlas.tl.PalantirExtension.run` expects as its starting point. Selection is
        delegated to :func:`palantir.utils.early_cell`; this method adapts the
        ``MuData`` object to the :class:`~anndata.AnnData` that function requires.

        Parameters
        ----------
        celltype
            Cell type to select the initial cell from.
        cluster_key
            Column in ``mudata.obs`` holding the cell type annotation.
        eigvec_multi_key
            Key in ``mudata.obsm`` where the multiscale space is stored.
        fallback_seed
            Enables the fallback described in the Notes when no cell of ``celltype``
            lies at an extreme. If ``None`` (default), that situation raises instead.
        random_state
            Random seed used if the multiscale space has to be computed first.

        Returns
        -------
        Barcode of the selected initial cell, suitable for :meth:`~atlas.tl.PalantirExtension.run`.

        Raises
        ------
        KeyError
            If ``cluster_key`` is not a column of ``mudata.obs``.
        ValueError
            If ``celltype`` does not occur in ``mudata.obs[cluster_key]``.
        palantir.utils.CellNotFoundException
            If no cell of ``celltype`` lies at an extreme of the multiscale space and
            ``fallback_seed`` is ``None``. The message describes how to proceed.

        Warns
        -----
        UserWarning
            Before the fallback runs, since it is as expensive as :meth:`~atlas.tl.PalantirExtension.run` itself.

        Notes
        -----
        If the multiscale space identified by ``eigvec_multi_key`` is not present, the
        anisotropic kernel, the diffusion components and the multiscaled distances are
        inferred using the default parameters, as :meth:`~atlas.tl.PalantirExtension.run` also does.

        Only a limited number of cells are candidates, namely both extremes of each
        diffusion component, so a cell type that does not occupy an extreme will not be
        found. Increasing the number of diffusion components is usually the cheaper
        remedy.

        Setting ``fallback_seed`` enables
        :func:`palantir.utils.fallback_terminal_cell`, which runs Palantir in full from
        an arbitrarily chosen cell of another type and returns the cell of ``celltype``
        furthest along the resulting pseudotime. The result is reproducible for a given
        seed but depends on that arbitrary choice.

        That internal run takes Palantir's own defaults, which coincide with the defaults
        of :meth:`atlas.tl.PalantirExtension.run` for ``knn``, ``num_waypoints``, ``n_jobs``, ``scale_components``
        and ``max_iterations``, but not for the random seed. They cannot be set from here,
        so a caller who passes non-default values to :meth:`atlas.tl.PalantirExtension.run` should not expect the
        fallback to match them. The multiscale space is guaranteed to be the one ATLAS
        uses, and nothing computed during the fallback is written to the ``MuData``.

        """
        if cluster_key not in self._mudata.obs.columns:
            raise KeyError(f"{cluster_key} not in mudata.obs")

        if celltype not in self._mudata.obs[cluster_key].values:
            raise ValueError(f"Cell type '{celltype}' not found in mudata.obs['{cluster_key}']")

        # If multiscale distances are not present they are automatically computed
        # using standard parameters, matching the behaviour of `run`
        if eigvec_multi_key not in self._mudata.obsm:
            self.compute_kernel()
            self.compute_diffusion_maps(seed=random_state)
            self.compute_multiscale_space(out_key=eigvec_multi_key)

        keys = list(dict.fromkeys([eigvec_multi_key, _PALANTIR_FALLBACK_EIGVEC_KEY]))
        _tmp = _palantir_anndata(self._mudata, eigvec_keys=keys, multiscale=self._mudata.obsm[eigvec_multi_key])

        # Attempt the inexpensive scan first so the fallback is only announced when it
        # is actually reached, rather than whenever `fallback_seed` happens to be set.
        try:
            return palantir.utils.early_cell(
                _tmp, celltype=celltype, celltype_column=cluster_key, eigvec_key=eigvec_multi_key, fallback_seed=None
            )
        except palantir.utils.CellNotFoundException:
            if fallback_seed is None:
                raise
            warnings.warn(
                f"No cell of type '{celltype}' lies at an extreme of the multiscale space. "
                "Falling back to running Palantir in full from an arbitrary anchor cell, solely "
                "to select one initial cell: this costs as much as `run` itself and its result is "
                "discarded apart from the returned barcode. Increasing the number of diffusion "
                "components is usually the cheaper remedy.",
                stacklevel=2,
            )
            return palantir.utils.early_cell(
                _tmp,
                celltype=celltype,
                celltype_column=cluster_key,
                eigvec_key=eigvec_multi_key,
                fallback_seed=fallback_seed,
            )

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
            Column of ``mudata.obs`` whose values name the inferred states if specified
            , instead of :cite:`palantir` default of naming them by barcode.
            Where several states would take one name they
            are disambiguated with a ``_suffix``.

            .. versionchanged:: 1.1.0
               Previously, terminal cells sharing a value of this column were merged into a
               single state and their fate probabilities added together, so this parameter
               changed how many fates were reported and what they were worth. It now only
               renames, and ``.obsm[fate_prob_key]`` has one column per terminal state.
        terminal_states
            Terminal states can be defined by the user. When ``None``,
            terminal states are automatically inferred by the algorithm.
        knn:
            Number of nearest neighbors for graph construction in the multiscale space.
        num_waypoints:
            Number of waypoints for trajectory inference. See the :cite:`palantir`
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
            - Fate probabilites stored as a :class:`pandas.DataFrame` in `obsm[fate_prob_key]`.
            - List of inferred waypoints in `.uns[waypoints_key]`.
            - States as categorical columns of ``.obs``: ``"initial_states"``, ``"terminal_states"``
              and ``"macrostates"``. A cell carries the name of the state it belongs to, or no
              value where it belongs to none. Spearate columns per kind are what let a cell belong
              to states of more than one kind.
            - One colour per state name, in ``.uns["initial_states_colors"]``,
              ``.uns["terminal_states_colors"]`` and ``.uns["macrostates_colors"]``, each aligned to
              its column's categories, and in ``.uns["atlas_state_palette"]``.
            - Initial states are stored in `.uns["initial_states"]`
            - Terminal states are stored in `uns["terminal_states"]`.
            - State-specific colors stored in ``.uns["fate_state_colors"]``.

              .. deprecated:: 1.1.0
                 The three ``.uns`` entries above are superseded by the columns and colour lists,
                 and will be removed in version 2.0.0. They are retained in version 1.1.0 for
                 backwards compatibility. A stored key cannot warn when it is read, so
                 :func:`~atlas.tl.migrate_states` converts an object saved by an earlier version.

            - Entropy measures (e.g. Shannon entropy, KL divergence) stored in ``.obs``.

        Notes
        -----
        Palantir computes no coarse-graining of its own, so ``.obs["macrostates"]`` records the
        union of the initial and terminal states.

        A terminal state here is a single landmark cell — the endpoint Palantir found for a
        branch — so the state columns are far sparser than the equivalent from
        :class:`~atlas.tl.CellRankExtension`, which designates a population of representative
        cells per state. That difference belongs to the methods, not to how they are recorded.

        If the multiscale space identified by ``eigvec_multi_key`` is not present,
        the anisotropc kernel, the diffusion components and the multiscaled distances are inferred
        using the default parameters on the graph stored as "wnn_distances" in ``.obsp``.

        """
        # If multiscale distances are not present they are automatically computed
        # using standard parameters
        if eigvec_multi_key not in self._mudata.obsm:
            self.compute_kernel()
            self.compute_diffusion_maps(seed=random_state)
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

        # Names inferred states after the cell type in .obs["cluster_key"] rather than by
        # Palantir's default, the barcode. This renames and nothing else: states are never
        # merged because the annotation gives them the same name, which would let naming
        # change how many fates are reported and what they are worth. Where a name is claimed
        # by more than one state it is disambiguated, as CellRank does.
        #
        # Disambiguation spans the union of the initial and terminal states — the namespace
        # `macrostates` covers — so a state belonging to both keeps one name across them.
        _states = dict.fromkeys([early_cell, *res.branch_probs.columns])
        if cluster_key is not None and cluster_key in self._mudata.obs.columns:
            cell_to_cluster = self._mudata.obs[cluster_key]
            _names = _disambiguate_names({cell: str(cell_to_cluster.loc[cell]) for cell in _states})
        else:
            _names = {cell: str(cell) for cell in _states}

        terminal_states = {_names[cell]: [cell] for cell in res.branch_probs.columns}
        initial_states = {_names[early_cell]: [early_cell]}
        fate_probs = res.branch_probs.rename(columns=_names)

        _initial = _states_to_column(initial_states, self._mudata.obs_names)
        _terminal = _states_to_column(terminal_states, self._mudata.obs_names)
        self._mudata.obs["initial_states"] = _initial
        self._mudata.obs["terminal_states"] = _terminal
        # Palantir computes no coarse-graining, so `macrostates` records the union of the
        # kinds it does produce, matching what `CellRankExtension` writes when it is given
        # its states rather than inferring them.
        self._mudata.obs["macrostates"] = _resolve_overlap(_initial, _terminal)

        self._mudata.uns["initial_states"] = initial_states
        self._mudata.uns["terminal_states"] = terminal_states
        self._mudata.obsm[fate_prob_key] = fate_probs
        _assign_state_colors(self._mudata)

        compute_entropy(mudata=self._mudata, fate_probability_key=fate_prob_key)
