import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from muon import MuData

import atlas

matplotlib.use("Agg")

SEED, CELLS = 42, 60

#: Carried by both modalities, so it can only be resolved by naming one of them.
SHARED_FEATURE = "GATA1"


@pytest.fixture
def mudata(embedding_mudata) -> MuData:
    return embedding_mudata()


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


# --------------------------------------------------------------------------------------
# naming the embedding
# --------------------------------------------------------------------------------------


def test_an_embedding_belonging_to_one_modality_can_be_named(mudata):
    """``"<modality>:<basis>"`` resolves against that modality rather than the object."""
    mudata.mod["rna"].obsm["X_pca"] = np.random.default_rng(SEED).random((CELLS, 2))

    ax = atlas.pl.embedding(mudata, basis="rna:X_pca", color="pseudotime", show=False)

    assert isinstance(ax, Axes)
    assert ax.collections[0].get_offsets().shape == (CELLS, 2)


def test_an_object_carrying_no_embedding_at_all_names_the_step_that_produces_one():
    """Distinct from naming an absent one: there is nothing to suggest instead."""
    from anndata import AnnData

    rna = AnnData(np.zeros((4, 2), dtype=np.float32))
    rna.obs_names = [f"c{i}" for i in range(4)]
    bare = MuData({"rna": rna})

    with pytest.raises(KeyError, match="carries no embedding"):
        atlas.pl.embedding(bare, color="anything", show=False)
