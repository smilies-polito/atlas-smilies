import warnings

import numpy as np
from muon import MuData
from pygam import LinearGAM, s


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """
    Compute weighted quantile.

    Parameters
    ----------
    x : np.ndarray
        Values.
    w : np.ndarray
        Weights associated with `x`.
    q : float
        Quantile to compute, between 0 and 1.

    Returns
    -------
    float
        Weighted quantile value.
    """
    idx = np.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]
    cw = np.cumsum(w_sorted)
    cw /= cw[-1]
    return np.interp(q, cw, x_sorted)


class MultiBranchGAM:
    """
    Fit branch-specific Generalized Additive Models (GAMs) along pseudotime.

    This class models the relationship between pseudotime and:
    - gene expression (pTF)
    - gene activity

    separately for each lineage/branch using fate probabilities as weights.

    Parameters
    ----------
    mudata : MuData
        MuData object containing modalities and annotations.
    ptf : str or int
        Name or index of the transcription factor in the RNA modality.
    gene : str or int
        Name or index of the gene in the activity modality.
    time_key : str, default="pseudotime"
        Key in `mudata.obs` containing pseudotime values.
    fate_prob_key : str, default="fate_probabilities"
        Key in `mudata.obsm` containing fate probabilities per branch.
    n_splines : int, default=8
        Number of splines used in the GAM.

    Notes
    -----
    Expression and activity are standardized before fitting to ensure
    comparability across branches.
    """

    def __init__(
        self,
        mudata: MuData,
        ptf: str | int,
        gene: str | int,
        time_key: str = "pseudotime",
        fate_prob_key: str = "fate_probabilities",
        n_splines: int = 8,
    ):
        self.mudata = mudata

        if ptf not in mudata.mod["rna"].var_names:
            raise KeyError(f"{ptf} not a valid pTF")
        self.ptf = ptf

        if gene not in mudata.mod["activity"].var_names:
            raise KeyError(f"{gene} not a valid gene")
        self.gene = gene

        self.n_splines = n_splines

        if time_key not in mudata.obs.columns:
            raise KeyError(f"{time_key} not a valid key.")
        self.time_key = time_key

        if fate_prob_key not in mudata.obsm.keys():
            raise KeyError(f"{fate_prob_key} not a valid key.")
        self.fate_key = fate_prob_key

        self.prepare_data(mudata=mudata, fate_prob_key=fate_prob_key, ptf=ptf, gene=gene)

    @property
    def predictions(self):
        """Predictions on an user defined number of nodes"""
        return self._predictions

    @property
    def models(self):
        """Models for ptf and gene along the branches"""
        return self._models

    def prepare_data(self, mudata: MuData, fate_prob_key: str, ptf: int | str, gene: int | str, eps: float = 1e-6):
        """
        Extract and preprocess data from MuData.

        Parameters
        ----------
        mudata : MuData
            Input MuData object.
        fate_prob_key : str
            Key for fate probabilities in `mudata.obsm`.
        ptf : str or int
            Transcription factor identifier in RNA modality.
        gene : str or int
            Gene identifier in activity modality.
        eps : float, default=1e-6
            Small constant to avoid division by zero during standardization.

        Notes
        -----
        - Extracts pseudotime, fate probabilities, and feature values.
        - Standardizes gene expression and activity.
        """
        self.pseudotime = mudata.obs[self.time_key].values.reshape(-1, 1)

        fate_df = mudata.obsm[self.fate_key]
        self.fate_prob = fate_df.values
        self.branch_names = fate_df.columns

        gex = mudata.mod["rna"][:, ptf].X.toarray().ravel()
        act = mudata.mod["activity"][:, gene].X.toarray().ravel()

        # standardization: comparability among curves
        self.gex = (gex - gex.mean()) / (gex.std() + eps)
        self.act = (act - act.mean()) / (act.std() + eps)

    def fit(self) -> None:
        """
        Fit GAM models for each branch.

        For each branch, fits two weighted GAMs:
        - one for gene expression (`gex`)
        - one for gene activity (`act`)

        The weights correspond to fate probabilities.

        Notes
        -----
        Branches with negligible total weight are skipped and a warning is issued.
        """
        models = {}
        for i, branch in enumerate(self.branch_names):
            weights = self.fate_prob[:, i]
            if np.sum(weights) < 1e-8:
                warnings.warn(f"WARNING: no cells transitioning towards {branch}, skipping", stacklevel=2)
                continue

            eGam = LinearGAM(s(0, n_splines=self.n_splines)).fit(self.pseudotime, self.gex, weights=weights)
            aGam = LinearGAM(s(0, n_splines=self.n_splines)).fit(self.pseudotime, self.act, weights=weights)
            models[branch] = {"weights": weights, "gam_exp": eGam, "gam_act": aGam}

        self._models = models

    def predict(self, n_points: int = 200, q_low: float = 0.02, q_high: float = 0.98, ci: float = 0.95) -> None:
        """
        Predict smooth trajectories and confidence intervals.

        Parameters
        ----------
        n_points : int, default=200
            Number of points in the pseudotime grid.
        q_low : float, default=0.02
            Lower quantile for pseudotime range (weighted).
        q_high : float, default=0.98
            Upper quantile for pseudotime range (weighted).
        ci : float, default=0.95
            Confidence interval width.

        Notes
        -----
        - Predictions are computed on a branch-specific pseudotime range.
        - Confidence intervals are obtained from the fitted GAM.

        Results are stored in `self.predictions`.
        """
        results = {}
        t = self.pseudotime.ravel()

        for branch, model in self._models.items():
            weights = model["weights"]
            t_min = _weighted_quantile(t, weights, q_low)
            t_max = _weighted_quantile(t, weights, q_high)
            t_grid = np.linspace(t_min, t_max, n_points).reshape(-1, 1)

            gex = model["gam_exp"].predict(t_grid)
            gex_ci = model["gam_exp"].confidence_intervals(t_grid, width=ci)
            act = model["gam_act"].predict(t_grid)
            act_ci = model["gam_act"].confidence_intervals(t_grid, width=ci)

            results[branch] = {
                "t_grid": t_grid.ravel(),
                "gex": gex,
                "gex_lower": gex_ci[:, 0],
                "gex_upper": gex_ci[:, 1],
                "act": act,
                "act_lower": act_ci[:, 0],
                "act_upper": act_ci[:, 1],
            }

        self._predictions = results
