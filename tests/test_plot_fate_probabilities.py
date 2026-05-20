import numpy as np
import pandas as pd
from anndata import AnnData
from muon import MuData

from atlas.pl import plot_fate_probabilities

SEED = 42
N_TRUNK = 40
N_BRANCH = 30  # cells per branch
N_CELLS = N_TRUNK + N_BRANCH * 2


def _make_umap(rng):
    jitter = 0.03
    trunk_t = np.linspace(1.0, 0.0, N_TRUNK)
    trunk = np.column_stack([rng.normal(0.0, jitter, size=N_TRUNK), trunk_t])

    lefts = np.linspace(0.0, 1.0, N_BRANCH)
    left = np.column_stack(
        [-lefts + rng.normal(0.0, jitter, size=N_BRANCH), -lefts + rng.normal(0.0, jitter, size=N_BRANCH)]
    )

    rights = np.linspace(0.0, 1.0, N_BRANCH)
    right = np.column_stack(
        [rights + rng.normal(0.0, jitter, size=N_BRANCH), -rights + rng.normal(0.0, jitter, size=N_BRANCH)]
    )

    return np.vstack([trunk, left, right]).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    obs = pd.DataFrame(index=obs_names)

    rna = AnnData(X=np.empty((N_CELLS, 0), dtype=np.float32), obs=obs.copy())
    activity = AnnData(X=np.empty((N_CELLS, 0), dtype=np.float32), obs=obs.copy())
    mudata = MuData({"rna": rna, "activity": activity})

    trunk_pt = np.linspace(0.0, 0.4, N_TRUNK)
    left_pt = np.linspace(0.4, 1.0, N_BRANCH)
    right_pt = np.linspace(0.4, 1.0, N_BRANCH)
    pseudotime = np.concatenate([trunk_pt, left_pt, right_pt]).astype(np.float32)
    mudata.obs["pseudotime"] = pseudotime
    mudata.obsm["X_umap"] = _make_umap(rng)

    # single fate scenario
    data = mudata.copy()
    terminal_cell = data.obs_names[-1]
    fate_probabilities = pd.DataFrame(np.ones((len(data), 1)), index=data.obs_names, columns=["terminal"])
    data.obsm["fate_probabilities"] = fate_probabilities
    data.uns["fate_state_colors"] = {"terminal": "#ff0000"}

    plot_fate_probabilities(mudata=data, embedding_key="X_umap", fate_probability_key="fate_probabilities")

    # two fates scenario
    data = mudata.copy()
    left_prob = np.zeros(N_CELLS, dtype=np.float32)
    right_prob = np.zeros(N_CELLS, dtype=np.float32)

    left_prob[:N_TRUNK] = 0.5
    right_prob[:N_TRUNK] = 0.5

    left_commitment = (left_pt - 0.4) / (1.0 - 0.4)
    right_commitment = (right_pt - 0.4) / (1.0 - 0.4)

    left_commitment = left_commitment**2
    right_commitment = right_commitment**2

    left_start = N_TRUNK
    left_end = N_TRUNK + N_BRANCH

    left_prob[left_start:left_end] = 0.5 + 0.5 * left_commitment
    right_prob[left_start:left_end] = 1.0 - left_prob[left_start:left_end]
    right_start = left_end

    right_prob[right_start:] = 0.5 + 0.5 * right_commitment
    left_prob[right_start:] = 1.0 - right_prob[right_start:]

    fate_probabilities = pd.DataFrame(
        {
            "left": left_prob,
            "right": right_prob,
        },
        index=data.obs_names,
    )

    data.obsm["fate_probabilities"] = fate_probabilities
    data.uns["fate_state_colors"] = {
        "left": "#1f77b4",
        "right": "#ff7f0e",
    }

    plot_fate_probabilities(mudata=data, embedding_key="X_umap", fate_probability_key="fate_probabilities")

    # committed totally to one cluster
    data = mudata.copy()
    left_prob = np.zeros(N_CELLS, dtype=np.float32)
    right_prob = np.zeros(N_CELLS, dtype=np.float32)
    fate_probabilities = pd.DataFrame({"left": left_prob, "right": right_prob}, index=obs_names)

    cells = rng.choice(mudata.obs_names, size=30, replace=False)
    right_cells = mudata.obs_names.difference(cells)
    fate_probabilities.loc[cells, "left"] = 1
    fate_probabilities.loc[right_cells, "right"] = 1

    data.obsm["fate_probabilities"] = fate_probabilities
    data.uns["fate_state_colors"] = {
        "left": "#1f77b4",
        "right": "#ff7f0e",
    }
    plot_fate_probabilities(mudata=data, embedding_key="X_umap", fate_probability_key="fate_probabilities")

    # one cluster fully committed the other not
    data = mudata.copy()
    left_prob = np.zeros(N_CELLS, dtype=np.float32)
    right_prob = np.zeros(N_CELLS, dtype=np.float32)
    left_prob[:N_TRUNK] = 0.0
    right_prob[:N_TRUNK] = np.linspace(0.2, 0.6, N_TRUNK)

    left_start = N_TRUNK
    left_end = N_TRUNK + N_BRANCH

    left_prob[left_start:left_end] = 1.0
    right_prob[left_start:left_end] = 0.0

    right_start = left_end

    left_prob[right_start:] = 0.0
    right_prob[right_start:] = np.linspace(0.6, 1.0, N_BRANCH)

    fate_probabilities = pd.DataFrame(
        {
            "left": left_prob,
            "right": right_prob,
        },
        index=data.obs_names,
    )

    data.obsm["fate_probabilities"] = fate_probabilities
    data.uns["fate_state_colors"] = {
        "left": "#1f77b4",
        "right": "#ff7f0e",
    }

    plot_fate_probabilities(mudata=data, embedding_key="X_umap", fate_probability_key="fate_probabilities")

    # three fate scenario: left, trunk, right
    data = mudata.copy()

    left_prob = np.zeros(data.n_obs, dtype=np.float32)
    trunk_prob = np.zeros(data.n_obs, dtype=np.float32)
    right_prob = np.zeros(data.n_obs, dtype=np.float32)

    # trunk: far from the junction = committed to trunk
    # in your UMAP, top trunk is far from the junction
    trunk_prob[:N_TRUNK] = np.linspace(1.0, 0.34, N_TRUNK)
    left_prob[:N_TRUNK] = (1.0 - trunk_prob[:N_TRUNK]) / 2.0
    right_prob[:N_TRUNK] = (1.0 - trunk_prob[:N_TRUNK]) / 2.0

    # left branch: fully committed to left
    left_start = N_TRUNK
    left_end = N_TRUNK + N_BRANCH

    left_prob[left_start:left_end] = 1.0
    trunk_prob[left_start:left_end] = 0.0
    right_prob[left_start:left_end] = 0.0

    # right branch: commitment to right increases away from the junction
    right_start = left_end

    right_prob[right_start:] = np.linspace(0.34, 1.0, N_BRANCH)
    left_prob[right_start:] = (1.0 - right_prob[right_start:]) / 2.0
    trunk_prob[right_start:] = (1.0 - right_prob[right_start:]) / 2.0

    fate_probabilities = pd.DataFrame(
        {
            "left": left_prob,
            "trunk": trunk_prob,
            "right": right_prob,
        },
        index=data.obs_names,
    )

    data.obsm["fate_probabilities"] = fate_probabilities
    data.uns["fate_state_colors"] = {
        "left": "#1f77b4",
        "trunk": "#2ca02c",
        "right": "#ff7f0e",
    }
