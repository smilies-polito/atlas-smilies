import warnings
from collections.abc import Sequence

import numpy as np
from muon import MuData
from pygam import LinearGAM, s

#: Modalities the fit reads from: the transcription factor's expression and the genes'
#: chromatin activity. Both are required, there being nothing to relate without either.
_REQUIRED_MODALITIES = ("rna", "activity")


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Compute weighted quantile.

    A copy of the helper in :mod:`atlas.pl.utils`, which cannot be imported here: `atlas.pl`
    imports :class:`MultiLineageGAM` from `atlas.tl`, so reading back the other way would close
    a cycle. That copy serves the superseded fitting procedure and is removed with it in 2.0.0.

    Parameters
    ----------
    x
        Values.
    w
        Weights associated with `x`.
    q
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


class MultiLineageGAM:
    """Fit lineage-specific Generalized Additive Models along pseudotime.

    Uses :class:`pygam.pygam.LinearGAM`. Each lineage is modelled independently, with its
    fate probabilities employed as weights.

    Models the relationship between pseudotime and

    - the expression of one transcription factor, from the ``"rna"`` modality,
    - the activity of one or more genes, from the ``"activity"`` modality,

    separately for each lineage, using that lineage's fate probabilities as weights.

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying the modalities and annotations.
    ptf
        Name of the transcription factor in the ``"rna"`` modality.
    genes
        Name, or names, of the genes in the ``"activity"`` modality.
    time_key
        Key in ``mudata.obs`` containing pseudotime values.
    fate_prob_key
        Key in ``mudata.obsm`` containing the fate probabilities. Its columns name the
        lineages.
    n_splines
        Number of splines used in each GAM.

    Raises
    ------
    KeyError
        If the ``"rna"`` or ``"activity"`` modality is absent, if the factor or a gene is not
        found in the modality it is looked for in, or if ``time_key`` or ``fate_prob_key``
        names nothing.

    Notes
    -----
    Expression and activity are standardized before fitting so that curves are comparable
    across lineages.

    The transcription factor is fitted once per lineage however many genes are given. Its fit
    depends only on pseudotime, its own expression and that lineage's weights, so this is the
    same fit rather than an approximation of it.
    """

    def __init__(
        self,
        mudata: MuData,
        ptf: str,
        genes: str | Sequence[str],
        time_key: str = "pseudotime",
        fate_prob_key: str = "fate_probabilities",
        n_splines: int = 8,
    ):
        self.mudata = mudata

        # Checked before anything is looked up in them: the factor is read from `"rna"` and
        # the genes from `"activity"`, so a missing modality would otherwise surface as a bare
        # key error from the object naming neither what was wanted nor why.
        missing = [modality for modality in _REQUIRED_MODALITIES if modality not in mudata.mod]
        if missing:
            raise KeyError(f"missing required modalities in MuData: {', '.join(missing)}")

        if ptf not in mudata.mod["rna"].var_names:
            raise KeyError(f"{ptf} not a valid pTF")
        self.ptf = ptf

        genes = [genes] if isinstance(genes, str) else list(genes)
        for gene in genes:
            if gene not in mudata.mod["activity"].var_names:
                raise KeyError(f"{gene} not a valid gene")
        self.genes = genes

        self.n_splines = n_splines

        if time_key not in mudata.obs.columns:
            raise KeyError(f"{time_key} not a valid key.")
        self.time_key = time_key

        if fate_prob_key not in mudata.obsm.keys():
            raise KeyError(f"{fate_prob_key} not a valid key.")
        self.fate_key = fate_prob_key

        self.prepare_data(mudata=mudata, fate_prob_key=fate_prob_key, ptf=ptf, genes=genes)

    @property
    def predictions(self):
        """Predictions on a user defined number of points, keyed by lineage."""
        return self._predictions

    @property
    def models(self):
        """Fitted models for the factor and each gene, keyed by lineage."""
        return self._models

    def prepare_data(
        self, mudata: MuData, fate_prob_key: str, ptf: str, genes: Sequence[str], eps: float = 1e-6
    ) -> None:
        """Extract and standardize the data the models are fitted on.

        Parameters
        ----------
        mudata
            Multimodal annotated data object.
        fate_prob_key
            Key in ``mudata.obsm`` holding the fate probabilities.
        ptf
            Transcription factor in the ``"rna"`` modality.
        genes
            Genes in the ``"activity"`` modality.
        eps
            Small constant avoiding division by zero during standardization.
        """
        self.pseudotime = mudata.obs[self.time_key].values.reshape(-1, 1)

        fate_df = mudata.obsm[self.fate_key]
        self.fate_prob = fate_df.values
        self.lineage_names = fate_df.columns

        gex = mudata.mod["rna"][:, ptf].X.toarray().ravel()

        # standardization: comparability among curves. Each gene uses its own values only.
        self.gex = (gex - gex.mean()) / (gex.std() + eps)
        self.act = {}
        for gene in genes:
            act = mudata.mod["activity"][:, gene].X.toarray().ravel()
            self.act[gene] = (act - act.mean()) / (act.std() + eps)

    def fit(self) -> None:
        """Fit the models for each lineage.

        For every lineage, fits one weighted GAM for the factor's expression and one for each
        gene's activity. The weights are that lineage's fate probabilities.

        Notes
        -----
        Lineages with negligible total weight are skipped and a warning is issued.
        """
        models = {}
        for i, lineage in enumerate(self.lineage_names):
            weights = self.fate_prob[:, i]
            if np.sum(weights) < 1e-8:
                warnings.warn(f"WARNING: no cells transitioning towards {lineage}, skipping", stacklevel=2)
                continue

            eGam = LinearGAM(s(0, n_splines=self.n_splines)).fit(self.pseudotime, self.gex, weights=weights)
            gams_act = {
                gene: LinearGAM(s(0, n_splines=self.n_splines)).fit(self.pseudotime, act, weights=weights)
                for gene, act in self.act.items()
            }
            models[lineage] = {"weights": weights, "gam_exp": eGam, "gams_act": gams_act}

        self._models = models

    def predict(self, n_points: int = 200, q_low: float = 0.02, q_high: float = 0.98, ci: float = 0.95) -> None:
        """Predict smooth trajectories and confidence intervals.

        Parameters
        ----------
        n_points
            Number of points in the pseudotime grid.
        q_low
            Lower weighted quantile bounding the lineage's pseudotime range.
        q_high
            Upper weighted quantile bounding the lineage's pseudotime range.
        ci
            Width of the confidence interval.

        Notes
        -----
        The grid belongs to the lineage, being derived from its weights, and is therefore
        shared by every gene drawn for it. Results are stored in
        :attr:`~atlas.tl.MultiLineageGAM.predictions`.
        """
        results = {}
        t = self.pseudotime.ravel()

        for lineage, model in self._models.items():
            weights = model["weights"]
            t_min = _weighted_quantile(t, weights, q_low)
            t_max = _weighted_quantile(t, weights, q_high)
            t_grid = np.linspace(t_min, t_max, n_points).reshape(-1, 1)

            gex = model["gam_exp"].predict(t_grid)
            gex_ci = model["gam_exp"].confidence_intervals(t_grid, width=ci)

            genes = {}
            for gene, gam in model["gams_act"].items():
                act = gam.predict(t_grid)
                act_ci = gam.confidence_intervals(t_grid, width=ci)
                genes[gene] = {"act": act, "act_lower": act_ci[:, 0], "act_upper": act_ci[:, 1]}

            results[lineage] = {
                "t_grid": t_grid.ravel(),
                "gex": gex,
                "gex_lower": gex_ci[:, 0],
                "gex_upper": gex_ci[:, 1],
                "genes": genes,
            }

        self._predictions = results
