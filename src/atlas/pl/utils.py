import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from muon import MuData
from pygam import LinearGAM, s

from atlas.tl.utils import _KEY_REMOVAL_VERSION, _LEGACY_PALETTE_KEY

_FATE_PRODUCER = "`atlas.tl.CellRankExtension.compute_fate_probabilities`"
_EMBEDDING_PRODUCER = "`atlas.tl.umap`"


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


def _state_colors(mudata: MuData, kind: str) -> dict[str, str]:
    column = mudata.obs.get(kind)
    colors = mudata.uns.get(f"{kind}_colors")

    if column is not None and isinstance(column.dtype, pd.CategoricalDtype) and colors is not None:
        categories = [str(name) for name in column.cat.categories]
        colors = list(colors)
        if len(colors) == len(categories):
            return dict(zip(categories, colors, strict=True))

    superseded = mudata.uns.get(_LEGACY_PALETTE_KEY)
    if isinstance(superseded, Mapping) and superseded:
        warnings.warn(
            f"`mudata.uns['{_LEGACY_PALETTE_KEY}']` is superseded by "
            f"`mudata.uns['{kind}_colors']` and will be removed in {_KEY_REMOVAL_VERSION}. It is "
            f"being read for this call. Run `atlas.tl.migrate_states` to record this object's "
            f"colours in the current form; a stored key cannot announce this when it is read, so "
            f"nothing else will.",
            FutureWarning,
            stacklevel=3,
        )
        return {str(name): color for name, color in superseded.items()}

    return {}


def _fate_frame(mudata: MuData, key: str) -> pd.DataFrame:
    """The recorded probabilities as a frame whose columns name the fates."""
    if key not in mudata.obsm:
        available = ", ".join(sorted(mudata.obsm)) or "nothing"
        raise KeyError(
            f"'{key}' not in mudata.obsm, which carries {available}; run {_FATE_PRODUCER} to compute fate probabilities"
        )

    recorded = mudata.obsm[key]
    if isinstance(recorded, pd.DataFrame):
        return recorded


def _resolve_basis(mudata: MuData, basis: str) -> None:
    if basis in mudata.obsm or f"X_{basis}" in mudata.obsm:
        return

    modality, _, _ = basis.partition(":")
    if modality in mudata.mod:
        return

    available = ", ".join(sorted(key for key in mudata.obsm if key not in mudata.mod))
    if not available:
        raise KeyError(
            f"'{basis}' not in mudata.obsm, which carries no embedding; run {_EMBEDDING_PRODUCER} to compute one"
        )

    raise KeyError(
        f"'{basis}' not in mudata.obsm; run {_EMBEDDING_PRODUCER} to compute an embedding, "
        f"or pass `basis` naming one of those present ({available})"
    )


def _in_modality(adata: AnnData, modality: str, key: str, use_raw: bool | None) -> bool:
    name = key.split(":", 1)[1] if key.startswith(f"{modality}:") else key

    if name in adata.var_names:
        return True

    return (use_raw is None or use_raw) and adata.raw is not None and name in adata.raw.var_names


def _resolve_color(mudata: MuData, color: str | Sequence[str], use_raw: bool | None) -> None:

    keys = [color] if isinstance(color, str) else list(color)

    for key in keys:
        if key in mudata.obs.columns:
            continue

        carriers = [m for m in mudata.mod if _in_modality(mudata.mod[m], m, key, use_raw)]

        if len(carriers) == 1:
            continue

        if len(carriers) > 1 and ":" not in key:
            raise KeyError(
                f"'{key}' is a feature of more than one modality ({', '.join(sorted(carriers))}); "
                f"pass `<modality>:{key}` naming the one to color by"
            )

        if carriers:
            continue

        searched = ", ".join(sorted(mudata.mod))
        raise KeyError(
            f"'{key}' is neither a column of mudata.obs nor a feature of any modality "
            f"(searched mudata.obs and the var_names of {searched}); "
            "pass `<modality>:<feature>` to name a feature of a particular modality"
        )
