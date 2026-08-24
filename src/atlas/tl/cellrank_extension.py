from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from cellrank.estimators import GPCCA
from cellrank.kernels import PseudotimeKernel
from muon import MuData

from .utils import (
    _assign_state_colors,
    _cellrank_anndata,
    _deprecated_key_arg,
    _intermediate_states,
    _invert_assignment,
    _resolve_graph_key,
    _resolve_overlap,
    compute_entropy,
)


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
        key: str | None = None,
        connectivity_key: str | None = None,
        time_key: str = "pseudotime",
        cluster_key: str | None = None,
        backward: bool = False,
        threshold_scheme: Literal["soft", "hard"] = "hard",
        frac_to_keep: float = 0.3,
        b: float = 10.0,
        nu: float = 0.5,
        n_jobs: int = -1,
        backend: str = "loky",
    ) -> None:
        # This function is adapted from CellRank (BSD 3-Clause License).
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

        .. deprecated:: 1.1.0
            The `connectivity_key` parameter is deprecated and
            will be removed in version 2.0.0. It is retained in version
            1.1.0 for backwards compatibility. Use `key` to adopt the new behavior.

        Parameters
        ----------
        key
            Key in ``mudata.uns`` where the graph is recorded. The connectivity matrix is
            resolved from it, so nothing else is needed to identify the graph. Defaults to
            ``"wnn"``.
        connectivity_key
            Key in ``mudata.obsp`` where the connectivity matrix corresponding to the
            WNN is stored.
            Deprecated in 1.1.0 and will be removed in version 2.0.0.
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
        backend
            Which backend to use for multiprocessing ('loky', 'multiprocessing', 'threading').

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
        _deprecated_key_arg("connectivity_key", connectivity_key, "key", key)

        if connectivity_key is None:
            connectivity_key = _resolve_graph_key(self._mudata, key if key is not None else "wnn", "connectivities")
        elif connectivity_key not in self._mudata.obsp.keys():
            raise KeyError(f"{connectivity_key} not in mudata.obsp")
        if time_key not in self._mudata.obs.columns:
            raise KeyError(f"{time_key} not in mudata.obs")
        if cluster_key is not None and cluster_key not in self.mudata.obs.columns:
            raise KeyError(f"{cluster_key} not in mudata.obs")

        self._time_key = time_key
        self._cluster_key = cluster_key

        _tmp = _cellrank_anndata(self.mudata, connectivity_key=connectivity_key, cluster_key=cluster_key)

        self._kernel = PseudotimeKernel(adata=_tmp, time_key=time_key, conn_key=connectivity_key, backward=backward)
        self._kernel.compute_transition_matrix(
            threshold_scheme=threshold_scheme, frac_to_keep=frac_to_keep, b=b, nu=nu, n_jobs=n_jobs, backend=backend
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
        n_initial_states: int | None = 1,
        solver: Literal["direct", "gmres", "lgmres", "bicgstab", "gcrotmk"] = "gmres",
        use_petsc: bool = True,
        n_jobs: int = -1,
        tol: float = 1e-6,
        preconditioner: str | None = None,
        backend: str = "loky",
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
            Method used to compute the Schur decomposition. ``"krylov"`` requires
            :mod:`petsc4py` and :mod:`slepc4py`, which ATLAS does not install; without them
            CellRank falls back to ``"brandts"``, which requires a dense transition matrix.
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
            PETSc must be obtained separately; without it CellRank falls back to SciPy.
            Under the default ``solver="gmres"`` that fallback stays sparse.
        n_jobs
            Number of parallel jobs for the iterative solver.
        tol
            Convergence tolerance for the iterative solver.
        preconditioner
            Optional preconditioner for the solver. See :meth:`cellrank.estimators.GPCCA.compute_fate_probabilities`.
        backend
            Which backend to use for multiprocessing ('loky', 'multiprocessing', 'threading').

        Returns
        -------
        Updates the original ``MuData`` object:

            - Fate probabilities stored as a :class:`pandas.DataFrame` in ``.obsm["fate_probabilities"]``.
            - States as categorical columns of ``.obs``: ``"initial_states"``, ``"terminal_states"``
              and ``"macrostates"``. A cell carries the name of the state it belongs to, or no
              value where it belongs to none. Separate columns per kind are what let a cell belong
              to states of more than one kind, which ``allow_overlap`` permits.
            - One colour per state name, in ``.uns["initial_states_colors"]``,
              ``.uns["terminal_states_colors"]`` and ``.uns["macrostates_colors"]``, each aligned to
              its column's categories, and in ``.uns["atlas_state_palette"]``.
            - Initial states stored in ``.uns["initial_states"]``.
            - Terminal states stored in ``.uns["terminal_states"]``.
            - Intermediate states (if inferred) stored in ``.uns["intermediate_states"]``.
            - State-specific colors stored in ``.uns["fate_state_colors"]``.

              .. deprecated:: 1.1.0
                 The four ``.uns`` entries above are superseded by the columns and colour lists,
                 and will be removed in version 2.0.0. They are retained in version 1.1.0 for
                 backwards compatibility. A stored key cannot warn when it is read, so
                 :func:`~atlas.tl.migrate_states` converts an object saved by an earlier version.

            - Entropy measures (e.g. Shannon entropy, KL divergence) stored in ``.obs``.

        Raises
        ------
        AttributeError
            If the kernel has not been computed prior to calling this method.

        Notes
        -----
        This implementation relies on :class:`cellrank.estimators.GPCCA` to perform
        coarse-graining of the Markov chain and infer lineage relationships.

        Optional PETSc acceleration. Two parameters can draw on :mod:`petsc4py` and
        :mod:`slepc4py`, which ATLAS does not install.

        The two parameters degrade differently when the solver is absent, and only one of
        them costs anything:

        - ``method="krylov"`` needs the solver. Without it CellRank uses ``"brandts"``,
          which requires a dense transition matrix.
        - ``use_petsc=True`` does not need it. CellRank falls back to SciPy, and under the
          default ``solver="gmres"`` that route is sparse, so the result and the memory
          profile are unchanged.

        If both ``initial_states`` and ``terminal_states`` are provided, no automatic
        state inference is performed. Otherwise, macrostates and lineage-driving states
        are inferred from the data.

        ``.obs["macrostates"]`` carries a genuine coarse-graining only where one was computed,
        which is when the states are inferred. Where they are supplied, ``compute_macrostates``
        is never called and the column records the union of the states given instead.

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
            solver=solver, use_petsc=use_petsc, n_jobs=n_jobs, tol=tol, preconditioner=preconditioner, backend=backend
        )

        self.mudata.obsm["fate_probabilities"] = pd.DataFrame(
            _G.fate_probabilities.X, index=self.mudata.obs_names, columns=_G.fate_probabilities.names
        )

        _initial, _terminal = _G.initial_states, _G.terminal_states
        self.mudata.obs["initial_states"] = _initial
        self.mudata.obs["terminal_states"] = _terminal

        if terminal_states is None and initial_states is None:
            self.mudata.obs["macrostates"] = _G.macrostates
        else:
            self.mudata.obs["macrostates"] = _resolve_overlap(_initial, _terminal)

        self.mudata.uns["initial_states"] = _invert_assignment(_initial)
        self.mudata.uns["terminal_states"] = _invert_assignment(_terminal)
        self.mudata.uns["intermediate_states"] = _intermediate_states(self.mudata)

        self._fate_key = "fate_probabilities"
        _assign_state_colors(self.mudata)

        compute_entropy(mudata=self._mudata, fate_probability_key="fate_probabilities")
