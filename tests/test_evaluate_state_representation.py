"""The metrics that read no state record are unaffected by the change of representation.

`terminal_pseudotime_enrichment` is the only entry point in `atlas.tl.evaluate` that reads a
state record, and it has its own module. The four here read `.obs` columns and
`.obsm["fate_probabilities"]`, which is not superseded, so converting an object must leave
every value they return exactly where it was. That is what these cases pin.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.tl import (
    fate_concentration_index,
    migrate_states,
    pearson_correlation,
    spearman_correlation,
    terminal_state_silhouette,
)

SEED = 42
N_CELLS, N_GENES = 60, 10
CELLS = [f"cell{i}" for i in range(N_CELLS)]
N_RESAMPLES = 200  # the metrics default to 10000; the value under test is not the CI


@pytest.fixture
def mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((N_CELLS, N_GENES)), obs=pd.DataFrame([], index=CELLS))
    act = AnnData(rng.random((N_CELLS, N_GENES)), obs=pd.DataFrame([], index=CELLS))
    mdata = MuData({"rna": rna, "activity": act})

    mdata.obs["pseudotime"] = np.linspace(0, 1, N_CELLS)
    mdata.obs["other"] = rng.random(N_CELLS)

    fates = rng.random((N_CELLS, 2))
    fates /= fates.sum(axis=1, keepdims=True)
    mdata.obsm["fate_probabilities"] = pd.DataFrame(fates, index=CELLS, columns=["A", "B"])

    # the superseded layout, as an object saved before the current one carries it
    mdata.uns["initial_states"] = {"root": CELLS[:5]}
    mdata.uns["terminal_states"] = {"A": CELLS[-10:-5], "B": CELLS[-5:]}
    mdata.uns["intermediate_states"] = {}
    mdata.uns["fate_state_colors"] = {"root": "#e41a1c", "A": "#377eb8", "B": "#4daf4a"}
    return mdata


def _scores(mdata: MuData) -> dict[str, float]:
    return {
        "pearson": pearson_correlation(mdata, "pseudotime", "other", n_resamples=N_RESAMPLES)[0],
        "spearman": spearman_correlation(mdata, "pseudotime", "other", n_resamples=N_RESAMPLES)[0],
        "concentration": fate_concentration_index(mdata, n_resamples=N_RESAMPLES)[0],
        "silhouette": terminal_state_silhouette(mdata, soft_assignment=True),
    }


def test_converting_the_object_does_not_move_any_of_them(mudata):
    before = _scores(mudata)
    migrate_states(mudata)
    after = _scores(mudata)

    assert after == before


def test_they_need_no_state_record_at_all(mudata):
    """None of the four consults a state record, so removing every one leaves them working."""
    for key in ("initial_states", "terminal_states", "intermediate_states"):
        del mudata.uns[key]

    scores = _scores(mudata)

    assert all(np.isfinite(value) for value in scores.values())


def test_they_do_not_announce_the_superseded_layout(mudata):
    """Only the entry point that reads a superseded record warns about one."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _scores(mudata)
