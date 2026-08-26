import warnings

import numpy as np
import pandas as pd
from muon import MuData
from scipy.spatial.distance import cdist
from scipy.stats import bootstrap, pearsonr, permutation_test, spearmanr

from .utils import _states_mapping


def pearson_correlation(
    mudata: MuData, key1: str, key2: str, seed: int = 42, n_resamples: int = 10000, confidence_level: float = 0.95
) -> tuple:
    """
    Compute the Pearson correlation coefficient between two variables stored in ``mudata.obs``.

    Parameters
    ----------
    mudata : MuData
        Multimodal annotated data object containing observations in ``.obs``.
    key1 : str
        Column name in ``mudata.obs`` for the first variable.
    key2 : str
        Column name in ``mudata.obs`` for the second variable.
    seed : int, optional (default: 42)
        Random seed used for bootstrap resampling.
    n_resamples : int, optional (default: 10000)
        Number of resampling iterations for bootstrap estimation.
    confidence_level : float, optional (default: 0.95)
        Confidence level for the confidence interval.

    Returns
    -------
    statistic : float
        Pearson correlation coefficient.
    pvalue : float
        Two-sided p-value associated with the correlation.
    ci : tuple[float, float] | None
        Confidence interval as ``(low, high)``. Returns ``None`` if input contains NaNs.

    Warns
    -----
    UserWarning
        If input contains NaN values.
    UserWarning
        If the number of samples is smaller than 500 (bootstrap is used).
    UserWarning
        If the correlation is exactly ±1, the confidence interval may be undefined.

    Raises
    ------
    KeyError
        If either key1 and key2 are not in ``mudata.obs``.

    Notes
    -----
    Bootstrap-based confidence intervals are used for small sample sizes to improve robustness.

    """
    if key1 not in mudata.obs.columns:
        raise KeyError(f"{key1} not in .obs")

    if key2 not in mudata.obs.columns:
        raise KeyError(f"{key2} not in .obs")

    x = mudata.obs[key1]
    y = mudata.obs[key2]

    if x.isna().any() or y.isna().any():
        warnings.warn("NaNs are not a valid input, return NaNs.", stacklevel=2)
        return np.nan, np.nan, None

    x = x.reindex(y.index)
    res = pearsonr(x.values, y.values)
    if abs(res.statistic) == 1:
        warnings.warn("Correlation is ±1, confidence interval might be undefined", stacklevel=2)

    # Exact p-value with permutation and confidence interval via bootstrapping
    if len(x) < 500:

        def stat(a1: np.ndarray, a2: np.ndarray) -> float:
            return pearsonr(a1, a2).statistic

        warnings.warn("Less than 500 samples, using bootstrap CI.", stacklevel=2)
        ci = bootstrap(
            (x.values, y.values),
            stat,
            n_resamples=n_resamples,
            paired=True,
            alternative="two-sided",
            confidence_level=confidence_level,
            random_state=seed,
        )
        ci = (ci.confidence_interval.low, ci.confidence_interval.high)

    else:
        ci = res.confidence_interval(confidence_level=confidence_level)
        ci = (ci.low, ci.high)

    return (res.statistic, res.pvalue, ci)


def spearman_correlation(
    mudata: MuData, key1: str, key2: str, seed: int = 42, n_resamples: int = 10000, confidence_level: float = 0.95
) -> tuple:
    """
    Compute the Spearman rank correlation between two variables stored in ``mudata.obs``.

    Parameters
    ----------
    mudata
        Multimodal annotated data object containing observations in ``.obs``.
    key1
        Column name in ``mudata.obs`` for the first variable.
    key2
        Column name in ``mudata.obs`` for the second variable.
    seed
        Random seed used for permutation testing and bootstrap resampling.
    n_resamples
        Number of resampling iterations for permutation test and bootstrap.
    confidence_level
        Confidence level for the bootstrap confidence interval.

    Returns
    -------
    statistic : float
        Spearman correlation coefficient.
    pvalue : float
        Two-sided p-value. Computed via permutation test for small sample sizes,
        otherwise using the asymptotic approximation from ``scipy.stats.spearmanr``.
    ci : tuple[float, float] | None
        Confidence interval as ``(low, high)``. Returns ``None`` if input contains NaNs.

    Raises
    ------
        KeyError
            If key1 and key2 are not in ``mudata.obs``.

    Warns
    -----
    UserWarning
        If input contains NaN values.
    UserWarning
        If the number of samples is smaller than 500 (permutation test is used).
    UserWarning
        If the correlation is exactly ±1, the confidence interval may be undefined.

    Notes
    -----
    Spearman correlation as implemented in  :func:`scipy.stats.spearmanr` does not have a simple closed-form confidence interval,
    thereby bootstrap resampling is used to estimate uncertainty.

    Permutation test was used to provide more accurate pvalues as described in :func:`scipy.stats.spearmanr`.

    """

    def stat_func(a1: np.ndarray, a2: np.ndarray) -> float:
        return spearmanr(a1, a2).statistic

    if key1 not in mudata.obs.columns:
        raise KeyError(f"{key1} not in .obs")

    if key2 not in mudata.obs.columns:
        raise KeyError(f"{key2} not in .obs")

    x = mudata.obs[key1]
    y = mudata.obs[key2]

    if x.isna().any() or y.isna().any():
        warnings.warn("NaNs are not a valid input, return NaNs.", stacklevel=2)
        return np.nan, np.nan, None

    x = x.reindex(y.index)

    res = spearmanr(x.values, y.values)
    statistic = res.statistic

    if len(x) < 500:
        warnings.warn("Small sample size: using permutation test for pvalue.", stacklevel=2)
        perm = permutation_test(
            (x.values, y.values), stat_func, permutation_type="pairings", n_resamples=n_resamples, random_state=seed
        )
        pval = perm.pvalue
    else:
        pval = res.pvalue

    if abs(res.statistic) == 1:
        warnings.warn("Correlation is ±1, confidence interval might be undefined", stacklevel=2)

    ci = bootstrap(
        (x.values, y.values),
        stat_func,
        n_resamples=n_resamples,
        paired=True,
        alternative="two-sided",
        confidence_level=confidence_level,
        random_state=seed,
    )
    ci = (ci.confidence_interval.low, ci.confidence_interval.high)

    return (statistic, pval, ci)


def fate_concentration_index(
    mudata: MuData,
    fate_key: str = "fate_probabilities",
    time_key: str = "pseudotime",
    seed: int = 42,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
) -> tuple:
    r"""
    Compute the correlation between fate concentration and pseudotime.

    The fate concentration index is defined as the sum of squared fate
    probabilities for each observation:

    .. math::
        C_i = \sum_{j} p_{ij}^2

    where :math:`p_{ij}` represents the probability of cell :math:`i`
    belonging to fate :math:`j`. This metric captures how concentrated
    or committed a cell is toward specific fates (higher values indicate
    stronger commitment).

    The function computes the Spearman rank correlation between the
    concentration index and pseudotime, along with a p-value and a
    bootstrap confidence interval.

    For small sample sizes (< 500 observations), a permutation test is
    used to estimate the p-value, as suggested by :func:`scipy.stats.spearmanr`.

    Parameters
    ----------
    mudata
        Multi-modal annotated data object containing fate probabilities
        and pseudotime annotations.
    fate_key
        Key in ``mudata.obsm`` where the fate probability matrix is stored.
        Rows correspond to observations and columns to different fates.
    time_key
        Key in ``mudata.obs`` containing pseudotime values.
    seed
        Random seed used for permutation tests and bootstrap resampling.
    n_resamples
        Number of resampling iterations for permutation test and bootstrap.
    confidence_level
        Confidence level for the bootstrap confidence interval.

    Returns
    -------
    statistic : float
        Spearman correlation coefficient between fate concentration
        index and pseudotime.
    pvalue : float
        Two-sided p-value associated with the correlation. Computed via
        permutation test if sample size is small, otherwise from
        ``scipy.stats.spearmanr``.
    ci : tuple[float, float]
        Lower and upper bounds of the bootstrap confidence interval.
    concentration_index : pandas.Series
        Computed concentration index for each observation.

    Raises
    ------
    KeyError
        If ``fate_key`` is not found in ``mudata.obsm`` or
        ``time_key`` is not found in ``mudata.obs``.

    Warns
    -----
    UserWarning
        If the fate probability matrix is empty.
    UserWarning
        If the correlation is ±1, making the confidence interval unreliable.
    UserWarning
        If the sample size is small (< 500), triggering permutation-based
        p-value estimation.


    """

    def stat_func(x: np.ndarray, y: np.ndarray) -> float:
        return spearmanr(x, y).statistic

    def _concentration_index(x: pd.DataFrame) -> pd.Series:
        return (x**2).sum(axis=1)

    if fate_key not in mudata.obsm:
        raise KeyError(f"{fate_key} not in mudata.obsm")

    if time_key not in mudata.obs:
        raise KeyError(f"{time_key} not in mudata.obs")

    fates = mudata.obsm[fate_key]
    pseudotime = mudata.obs[time_key]
    pseudotime = pseudotime.reindex(fates.index)

    if fates.shape[1] == 0:
        warnings.warn("Empty fate probability matrix: returning NaNs", stacklevel=2)
        return (np.nan, np.nan, None, None)

    concentration_index = _concentration_index(fates)
    res = spearmanr(concentration_index.values, pseudotime.values)
    statistic = res.statistic

    if abs(statistic) == 1:
        warnings.warn("Correlation is ±1, confidence interval might be undefined", stacklevel=2)

    if len(mudata.obs_names) < 500:
        warnings.warn("Small sample size: using permutation test for p-value.", stacklevel=2)
        perm = permutation_test(
            (pseudotime.values, concentration_index.values),
            stat_func,
            permutation_type="pairings",
            n_resamples=n_resamples,
            alternative="two-sided",
            random_state=seed,
        )
        pval = perm.pvalue
    else:
        pval = res.pvalue

    ci = bootstrap(
        (pseudotime.values, concentration_index.values),
        stat_func,
        n_resamples=n_resamples,
        paired=True,
        alternative="two-sided",
        confidence_level=confidence_level,
        random_state=seed,
    )
    ci = (ci.confidence_interval.low, ci.confidence_interval.high)

    return (statistic, pval, ci, concentration_index)


def _hard_ai(f: pd.DataFrame, D: pd.DataFrame) -> pd.Series:
    """
    Compute the intra-terminal distance :math:`a_i` for each cell (hard assignment).

    Parameters
    ----------
    f : pandas.DataFrame
        Fate probabilities (cells × terminal states).
    D : pandas.DataFrame
        Pairwise distance matrix between cells.

    Returns
    -------
    pandas.Series
        Mean intra-terminal distance for each cell. Returns ``NaN`` if a cell
        is the only one assigned to its terminal state.
    """
    cell_ids = f.index
    ai = pd.Series(index=cell_ids, dtype=float)
    assignment = f.idxmax(axis="columns")
    for i in cell_ids:
        t = assignment.loc[i]
        same_terminal = assignment[assignment == t].index
        same_terminal = same_terminal.drop(i)
        ai[i] = np.nan if len(same_terminal) == 0 else D.loc[i, same_terminal].mean()
    return ai


def _soft_ai(f: pd.DataFrame, D: pd.DataFrame) -> pd.Series:
    """
    Compute the intra-terminal distance :math:`a_i` for each cell (soft assignment).

    Parameters
    ----------
    f : pandas.DataFrame
        Fate probabilities (cells × terminal states).
    D : pandas.DataFrame
        Pairwise distance matrix between cells.

    Returns
    -------
    pandas.Series
        Weighted intra-terminal distance for each cell. Returns ``NaN`` if
        weights sum to zero.
    """
    cell_ids = f.index
    ai = pd.Series(index=cell_ids, dtype=float)
    for i in cell_ids:
        w = f.loc[i].values @ f.values.T
        w[f.index.get_loc(i)] = 0
        ai[i] = np.nan if w.sum() == 0 else (np.sum(w * D.loc[i].values) / w.sum())
    return ai


def _hard_bi(f: pd.DataFrame, D: pd.DataFrame) -> pd.Series:
    """
    Compute the nearest-terminal distance :math:`b_i` for each cell (hard assignment).

    Parameters
    ----------
    f : pandas.DataFrame
        Fate probabilities (cells × terminal states).
    D : pandas.DataFrame
        Pairwise distance matrix between cells.

    Returns
    -------
    pandas.Series
        Minimum inter-terminal distance for each cell. Returns ``NaN`` if no
        alternative terminal states are available.
    """
    cells_ids, terminal = f.index, f.columns
    bi = pd.Series(index=cells_ids, dtype=float)
    assignment = f.idxmax(axis="columns")
    for i in cells_ids:
        ti = assignment.loc[i]
        b_candidates = []
        for t in terminal:
            if t == ti:
                continue
            cells_t = assignment[assignment == t].index
            if len(cells_t) == 0:
                continue
            b_t = D.loc[i, cells_t].mean()
            b_candidates.append(b_t)
        bi[i] = np.nan if len(b_candidates) == 0 else np.min(b_candidates)
    return bi


def _soft_bi(f: pd.DataFrame, D: pd.DataFrame) -> pd.Series:
    """
    Compute the nearest-terminal distance :math:`b_i` for each cell (soft assignment).

    Parameters
    ----------
    f : pandas.DataFrame
        Fate probabilities (cells × terminal states).
    D : pandas.DataFrame
        Pairwise distance matrix between cells.

    Returns
    -------
    pandas.Series
        Weighted inter-terminal distance for each cell. Returns ``NaN`` if
        weights sum to zero.
    """
    cell_ids = f.index
    bi = pd.Series(index=cell_ids, dtype=float)
    for i in cell_ids:
        w = f.loc[i].values @ f.values.T
        w[f.index.get_loc(i)] = 0
        one_w = 1.0 - w
        one_w[f.index.get_loc(i)] = 0
        denom = one_w.sum()
        bi[i] = np.nan if denom == 0 else np.sum(one_w * D.loc[i].values) / denom
    return bi


def terminal_state_silhouette(
    mudata: MuData,
    fate_key: str = "fate_probabilities",
    soft_assignment: bool = True,
    time_key: str | None = None,
    alpha: float = 1,
) -> float:
    r"""
    Compute the silhouette score for terminal states.

    This metric evaluates whether cells committed to the same lineage are more
    similar to each other than to cells belonging to other lineages. It extends
    the classical silhouette score to probabilistic fate assignments.

    Two strategies are supported to handle partially committed cells:

    - **Soft assignment**: cells are not assigned to a single lineage, and
      distances are weighted by fate probabilities.
    - **Pseudotime weighting**: silhouette scores are weighted by pseudotime,
      giving more importance to committed cells.

    The silhouette score is computed as:

    .. math::

        s_i = \frac{b_i - a_i}{\max(a_i, b_i)}

    where :math:`a_i` is the intra-terminal distance and :math:`b_i` is the
    nearest-terminal distance.

    When ``soft_assignment=False``, the final score is computed as:

    .. math::

        S = \frac{\sum_i \tau_i^\alpha s_i}{\sum_i \tau_i^\alpha}

    where :math:`\tau_i` is the pseudotime of cell :math:`i`.


    Parameters
    ----------
    mudata : MuData
        Annotated multimodal data object.
    fate_key
        Key in ``mudata.obsm`` containing fate probabilities
        (cells × terminal states).
    soft_assignment
        Whether to use the soft assignment strategy. If ``False``, hard
        assignment with pseudotime weighting is used.
    time_key
        Key in ``mudata.obs`` containing pseudotime values. Required when
        ``soft_assignment=False``.
    alpha
        Exponent used to weight pseudotime in the silhouette aggregation.

    Returns
    -------
    Terminal state silhouette score.

    """
    if fate_key not in mudata.obsm:
        raise KeyError(f"{fate_key} not in mudata.obsm")

    fates = mudata.obsm[fate_key]

    if fates.isna().any().any():
        raise ValueError("NaN values in fates are not supported")

    if fates.shape[1] == 0:
        raise ValueError("No terminal states available.")

    if fates.shape[1] == 1:
        warnings.warn("Only one terminal state available: silhouette is undefined, returning 0.", stacklevel=2)
        return 0

    D = pd.DataFrame(cdist(fates.values, fates.values, metric="euclidean"), index=fates.index, columns=fates.index)

    ai = _soft_ai(f=fates, D=D) if soft_assignment else _hard_ai(f=fates, D=D)
    bi = _soft_bi(f=fates, D=D) if soft_assignment else _hard_bi(f=fates, D=D)

    # no division by 0
    maximum = np.maximum(ai, bi)
    maximum = np.where(maximum == 0, 1e-6, maximum)

    si = (bi - ai) / maximum
    if soft_assignment:
        S = np.mean(si)
    else:
        if time_key is None or time_key not in mudata.obs.columns:
            raise KeyError(f"{time_key} must be an available mudata.obs column")
        pseudotime = mudata.obs[time_key]
        pseudotime = pseudotime.reindex(fates.index)
        S = np.sum((pseudotime**alpha) * si) / np.sum(pseudotime**alpha)

    return S


def terminal_pseudotime_enrichment(mudata: MuData, time_key: str = "pseudotime", rank: bool = False) -> float:
    r"""
    Compute the average pseudotime enrichment across terminal states.

    This function quantifies whether terminal states are enriched toward
    higher (or lower) pseudotime values. For each terminal state, it computes
    the difference between the median pseudotime of the cells in that state
    and the global median pseudotime across all cells:

    .. math::
        E_{ts} = \mathrm{median}(m_{ts}) - \mathrm{median}(m_{all})

    where :math:`m` is either the raw pseudotime or its normalized rank
    transformation. The final score is the average enrichment across all
    terminal states:

    .. math::
        E = \frac{1}{T} \sum_{ts} E_{ts}

    where :math:`T` is the number of terminal states.

    Positive values indicate that terminal states are enriched toward higher
    pseudotime (i.e., later stages), while negative values indicate enrichment
    toward lower pseudotime.

    Parameters
    ----------
    mudata
        Annotated multimodal dataset containing pseudotime values and
        terminal state annotations.
    time_key
        Key in ``mudata.obs`` where pseudotime values are stored.
    rank
        If True, pseudotime values are converted to normalized ranks in
        the interval [0, 1] before computing enrichment. This makes the
        score robust to non-linear scaling of pseudotime.

    Returns
    -------
    Mean enrichment score across all terminal states. Returns ``NaN``
    if enrichment cannot be computed (e.g., no valid cells in terminal
    states).

    Raises
    ------
    KeyError
        If ``time_key`` is not present in ``mudata.obs``.

    Notes
    -----
    Returns NaN if no terminal states are found.

    """
    if time_key not in mudata.obs:
        raise KeyError(f"{time_key} not in mudata.obs.")
    pseudotime = mudata.obs[time_key]

    terminal_states = _states_mapping(mudata, "terminal_states")
    if not bool(terminal_states):
        warnings.warn("No terminal states available, returning NaN", stacklevel=2)
        return np.nan

    if rank:
        m = pseudotime.rank(method="average")
        if len(m) > 1:
            m = (m - 1) / (len(m) - 1)
        else:
            m = pd.Series(0.0, index=pseudotime.index)
    else:
        m = pseudotime

    m_global = m.median()
    enrichment = pd.Series(index=terminal_states.keys(), dtype=float)
    for ts, cells in terminal_states.items():
        enrichment.loc[ts] = m.loc[cells].median() - m_global
    return enrichment.mean()
