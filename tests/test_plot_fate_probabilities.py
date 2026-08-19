"""The superseded entry point for drawing fate probabilities.

Covers :func:`atlas.pl.plot_fate_probabilities`, deprecated in 1.1.0 and removed in
2.0.0. Its contract for 1.x is that it announces its supersession and still guards its
inputs; :func:`atlas.pl.fate_probabilities` that supersedes it is covered in
``test_fate_probabilities.py``.

This module previously held no tests at all: its whole body sat under
``if __name__ == "__main__"``, so pytest collected nothing from it and every guard below
was unexercised.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

import atlas

matplotlib.use("Agg")

SEED = 42
N_TRUNK, N_BRANCH = 40, 30
N_CELLS = N_TRUNK + N_BRANCH * 2
FATES = ["left", "right"]
COLOURS = {"left": "#1f77b4", "right": "#ff7f0e"}


def _make_umap(rng):
    """A trunk that splits into two branches, so the drawing has something to shade."""
    jitter = 0.03
    trunk_t = np.linspace(1.0, 0.0, N_TRUNK)
    trunk = np.column_stack([rng.normal(0.0, jitter, size=N_TRUNK), trunk_t])

    steps = np.linspace(0.0, 1.0, N_BRANCH)
    left = np.column_stack(
        [-steps + rng.normal(0.0, jitter, size=N_BRANCH), -steps + rng.normal(0.0, jitter, size=N_BRANCH)]
    )
    right = np.column_stack(
        [steps + rng.normal(0.0, jitter, size=N_BRANCH), -steps + rng.normal(0.0, jitter, size=N_BRANCH)]
    )
    return np.vstack([trunk, left, right]).astype(np.float32)


@pytest.fixture
def mudata() -> MuData:
    """Two fates that commit along the two branches, with a colour recorded for each."""
    rng = np.random.default_rng(SEED)
    obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    obs = pd.DataFrame(index=obs_names)

    rna = AnnData(X=np.empty((N_CELLS, 0), dtype=np.float32), obs=obs.copy())
    activity = AnnData(X=np.empty((N_CELLS, 0), dtype=np.float32), obs=obs.copy())
    mudata = MuData({"rna": rna, "activity": activity})

    branch = np.linspace(0.4, 1.0, N_BRANCH)
    mudata.obs["pseudotime"] = np.concatenate([np.linspace(0.0, 0.4, N_TRUNK), branch, branch]).astype(np.float32)
    mudata.obsm["X_umap"] = _make_umap(rng)

    commitment = ((branch - 0.4) / 0.6) ** 2
    left = np.concatenate([np.full(N_TRUNK, 0.5), 0.5 + 0.5 * commitment, 0.5 - 0.5 * commitment])
    mudata.obsm["fate_probabilities"] = pd.DataFrame(
        {"left": left, "right": 1.0 - left}, index=obs_names, dtype=np.float32
    )
    mudata.uns["fate_state_colors"] = dict(COLOURS)
    return mudata


# --------------------------------------------------------------------------------------
# the supersession
# --------------------------------------------------------------------------------------


def test_the_superseded_entry_point_announces_its_supersession(mudata):
    with pytest.warns(FutureWarning) as record:
        atlas.pl.plot_fate_probabilities(mudata)

    messages = [str(warning.message) for warning in record]
    assert any("atlas.pl.fate_probabilities" in message and "2.0.0" in message for message in messages)


def test_the_superseded_entry_point_still_draws(mudata):
    with pytest.warns(FutureWarning):
        assert atlas.pl.plot_fate_probabilities(mudata) is None

    assert plt.get_fignums()


# --------------------------------------------------------------------------------------
# the guards the docstring documents
# --------------------------------------------------------------------------------------


def test_probabilities_that_are_not_recorded_warn_and_draw_nothing(mudata):
    del mudata.obsm["fate_probabilities"]

    with pytest.warns(UserWarning, match="fate probabilities are not available"):
        assert atlas.pl.plot_fate_probabilities(mudata) is None


def test_an_embedding_that_is_not_recorded_warns_and_draws_nothing(mudata):
    with pytest.warns(UserWarning, match="embedding is not available"):
        assert atlas.pl.plot_fate_probabilities(mudata, embedding_key="X_absent") is None


def test_one_fate_may_be_given_as_a_name(mudata):
    with pytest.warns(FutureWarning):
        assert atlas.pl.plot_fate_probabilities(mudata, states="left") is None

    assert plt.get_fignums()


def test_naming_only_fates_that_do_not_exist_selects_none_of_them(mudata):
    with pytest.warns(UserWarning, match="No lineages have been selected"):
        assert atlas.pl.plot_fate_probabilities(mudata, states=["absent"]) is None
