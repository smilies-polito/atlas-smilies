import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.axes import Axes
from muon import MuData

import atlas

matplotlib.use("Agg")

SEED, CELLS = 42, 60

#: Carried by both modalities, so it can only be resolved by naming one of them.
SHARED_FEATURE = "GATA1"


def _mudata() -> MuData:
    """Two modalities, two embeddings, and one `.obs` column of every kind that is drawn.

    The second embedding is deliberately one the wider ecosystem does not privilege by
    name: resolving it is the case the superseded entry point cannot do at all.
    """
    rng = np.random.default_rng(SEED)
    obs_names = [f"cell{i}" for i in range(CELLS)]

    rna = AnnData(rng.random((CELLS, 4)).astype(np.float32))
    rna.obs_names = obs_names
    rna.var_names = [SHARED_FEATURE, "rna_only", "g2", "g3"]

    activity = AnnData(rng.random((CELLS, 4)).astype(np.float32))
    activity.obs_names = obs_names
    activity.var_names = [SHARED_FEATURE, "activity_only", "a2", "a3"]

    mudata = MuData({"rna": rna, "activity": activity})

    time = np.linspace(0.0, 1.0, CELLS)
    mudata.obsm["X_umap"] = np.column_stack([time * np.cos(time * 6), time * np.sin(time * 6)])
    mudata.obsm["X_diffmap"] = rng.random((CELLS, 4))

    mudata.obs["pseudotime"] = time
    mudata.obs["n_counts"] = rng.integers(0, 5000, CELLS)
    mudata.obs["celltype"] = pd.Categorical(np.array(["HSC", "TAC", "Ery"])[rng.integers(0, 3, CELLS)])
    mudata.obs["labels"] = np.array(["left", "right"])[rng.integers(0, 2, CELLS)]
    mudata.obs["is_root"] = time < 0.3
    mudata.obs["with_missing"] = np.where(time < 0.5, np.nan, time)

    return mudata


@pytest.fixture
def mudata() -> MuData:
    return _mudata()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _has_colorbar(ax: Axes) -> bool:
    return ax.collections[0].colorbar is not None


def _has_legend(ax: Axes) -> bool:
    return ax.get_legend() is not None


# --------------------------------------------------------------------------------------
# the figure and what comes back
# --------------------------------------------------------------------------------------


def test_one_key_returns_its_axes(mudata):
    ax = atlas.pl.embedding(mudata, color="pseudotime", show=False)
    assert isinstance(ax, Axes)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_several_keys_return_one_axes_each(mudata):
    axes = atlas.pl.embedding(mudata, color=["pseudotime", "celltype", "n_counts"], show=False)
    assert isinstance(axes, list)
    assert len(axes) == 3
    assert all(isinstance(ax, Axes) for ax in axes)


def test_cells_are_drawn_without_a_colour(mudata):
    ax = atlas.pl.embedding(mudata, show=False)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_the_layout_is_controllable(mudata):
    axes = atlas.pl.embedding(mudata, color=["pseudotime", "celltype"], ncols=1, show=False)
    assert len(axes) == 2


# --------------------------------------------------------------------------------------
# resolving the embedding
# --------------------------------------------------------------------------------------


def test_the_default_is_where_the_embedding_step_writes(mudata):
    """`atlas.tl.umap` stores under ``X_umap``; drawing must need no argument to find it."""
    assert atlas.pl.embedding.__defaults__[0] == "umap"

    ax = atlas.pl.embedding(mudata, color="pseudotime", show=False)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_the_prefix_is_optional(mudata):
    bare = atlas.pl.embedding(mudata, basis="umap", color="pseudotime", show=False)
    prefixed = atlas.pl.embedding(mudata, basis="X_umap", color="pseudotime", show=False)

    assert np.array_equal(bare.collections[0].get_offsets(), prefixed.collections[0].get_offsets())


def test_an_embedding_outside_any_conventional_set_is_drawn(mudata):
    """The superseded entry point resolves through a hardcoded `pca`/`tsne`/`umap` list and
    fails here; the name alone is not grounds for refusing."""
    ax = atlas.pl.embedding(mudata, basis="X_diffmap", color="pseudotime", show=False)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_an_absent_embedding_names_the_step_that_produces_one(mudata):
    with pytest.raises(KeyError) as excinfo:
        atlas.pl.embedding(mudata, basis="X_nope", color="pseudotime", show=False)

    message = str(excinfo.value)
    assert "X_nope" in message
    assert "atlas.tl.umap" in message


def test_the_embeddings_offered_are_embeddings(mudata):
    """`.obsm` also carries a membership flag per modality; offering those as bases would
    name something that cannot be drawn."""
    with pytest.raises(KeyError) as excinfo:
        atlas.pl.embedding(mudata, basis="X_nope", show=False)

    message = str(excinfo.value)
    assert "X_umap" in message and "X_diffmap" in message
    assert "rna" not in message.replace("X_nope", "")


# --------------------------------------------------------------------------------------
# resolving what to colour by
# --------------------------------------------------------------------------------------


def test_an_observation(mudata):
    ax = atlas.pl.embedding(mudata, color="pseudotime", show=False)
    assert _has_colorbar(ax)


def test_a_feature_of_a_modality(mudata):
    """The superseded entry point rejects this: it resolves against `.obs` alone."""
    ax = atlas.pl.embedding(mudata, color="rna_only", show=False)
    assert _has_colorbar(ax)


def test_a_feature_qualified_by_its_modality(mudata):
    ax = atlas.pl.embedding(mudata, color="activity:activity_only", show=False)
    assert _has_colorbar(ax)


def test_a_name_several_modalities_carry_is_resolved_by_qualifying_it(mudata):
    for modality in ("rna", "activity"):
        ax = atlas.pl.embedding(mudata, color=f"{modality}:{SHARED_FEATURE}", show=False)
        assert _has_colorbar(ax)


def test_a_name_several_modalities_carry_is_refused_unqualified(mudata):
    """Left to the dependency this fails while joining the modalities' columns, reporting an
    overlap rather than the choice the caller has to make."""
    with pytest.raises(KeyError) as excinfo:
        atlas.pl.embedding(mudata, color=SHARED_FEATURE, show=False)

    message = str(excinfo.value)
    assert "rna" in message and "activity" in message
    assert f"<modality>:{SHARED_FEATURE}" in message


def test_a_key_naming_nothing_says_where_it_was_looked_for(mudata):
    with pytest.raises(KeyError) as excinfo:
        atlas.pl.embedding(mudata, color="nope", show=False)

    message = str(excinfo.value)
    assert "nope" in message
    assert "mudata.obs" in message
    assert "rna" in message and "activity" in message


def test_one_unresolvable_key_among_several_fails(mudata):
    with pytest.raises(KeyError):
        atlas.pl.embedding(mudata, color=["pseudotime", "nope"], show=False)


# --------------------------------------------------------------------------------------
# the kind of a key decides its treatment
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["pseudotime", "n_counts"])
def test_continuous_keys_get_a_colour_bar(mudata, key):
    ax = atlas.pl.embedding(mudata, color=key, show=False)
    assert _has_colorbar(ax)
    assert not _has_legend(ax)


@pytest.mark.parametrize("key", ["celltype", "labels", "is_root"])
def test_discrete_keys_get_a_legend(mudata, key):
    """Stored as an explicit category, as plain labels, and as a two-valued flag."""
    ax = atlas.pl.embedding(mudata, color=key, show=False)
    assert _has_legend(ax)
    assert not _has_colorbar(ax)


def test_both_kinds_in_one_call_render_by_their_own_kind(mudata):
    continuous, discrete = atlas.pl.embedding(mudata, color=["pseudotime", "celltype"], show=False)

    assert _has_colorbar(continuous) and not _has_legend(continuous)
    assert _has_legend(discrete) and not _has_colorbar(discrete)


def test_cells_with_no_value_are_still_drawn(mudata):
    ax = atlas.pl.embedding(mudata, color="with_missing", show=False)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_a_colour_map_with_a_discrete_key_is_disregarded_silently(mudata):
    """The superseded entry point passes its colormap whatever the key's kind, so every
    categorical call warns about a parameter the caller was entitled to supply."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        atlas.pl.embedding(mudata, color="celltype", cmap="viridis", show=False)

    assert not [w for w in caught if "colormapping" in str(w.message)]


# --------------------------------------------------------------------------------------
# discrete colours persist and are shared
# --------------------------------------------------------------------------------------


def test_recorded_colours_are_honoured(mudata):
    palette = ["#e41a1c", "#377eb8", "#4daf4a"]
    mudata.uns["celltype_colors"] = palette

    atlas.pl.embedding(mudata, color="celltype", show=False)

    assert list(mudata.uns["celltype_colors"]) == palette


def test_chosen_colours_are_recorded(mudata):
    assert "celltype_colors" not in mudata.uns

    atlas.pl.embedding(mudata, color="celltype", show=False)

    assert len(mudata.uns["celltype_colors"]) == len(mudata.obs["celltype"].cat.categories)


def test_the_same_key_keeps_its_colours(mudata):
    atlas.pl.embedding(mudata, color="celltype", show=False)
    first = list(mudata.uns["celltype_colors"])

    atlas.pl.embedding(mudata, color="celltype", show=False)

    assert list(mudata.uns["celltype_colors"]) == first


# --------------------------------------------------------------------------------------
# the superseded entry point is retained unchanged
# --------------------------------------------------------------------------------------


def test_the_superseded_entry_point_announces_its_supersession(mudata):
    with pytest.warns(FutureWarning) as record:
        atlas.pl.plot_embedding(mudata, show=False)

    message = str(record[0].message)
    assert "atlas.pl.embedding" in message
    assert "2.0.0" in message


def test_the_superseded_entry_point_still_returns_nothing(mudata):
    with pytest.warns(FutureWarning):
        assert atlas.pl.plot_embedding(mudata, show=False) is None


def test_the_superseded_entry_point_still_leaves_the_object_alone(mudata):
    """It draws on a throwaway object; recording colours is confined to the replacement."""
    before = set(mudata.obsm), set(mudata.uns)

    with pytest.warns(FutureWarning):
        atlas.pl.plot_embedding(mudata, observation="celltype", show=False)

    assert (set(mudata.obsm), set(mudata.uns)) == before
