import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.tl import PalantirExtension, migrate_states, reset_state_colors
from atlas.tl.utils import _intermediate_states, _resolve_overlap, _states_to_column

matplotlib.use("Agg")

SEED, K, NUM_WAYPOINTS = 42, 10, 20
CELLS = [f"cell{i}" for i in range(100)]
GENES = [f"gene{i}" for i in range(20)]
ACTIVITY_VAR = [f"act{i}" for i in range(15)]
EARLY_CELL = "cell0"

#: The three kinds of state, and the colour list each carries.
KINDS = ("initial_states", "terminal_states", "macrostates")


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _bare_mudata(n: int = 12) -> MuData:
    names = [f"c{i}" for i in range(n)]
    rna = AnnData(np.zeros((n, 2), dtype=np.float32))
    rna.obs_names = names
    return MuData({"rna": rna})


def _recorded_mudata() -> MuData:
    """States written directly, so the colour rules can be tested without an inference.

    ``t1`` is deliberately both an initial and a terminal state: that is the case the
    per-kind colour convention cannot handle on its own.
    """
    mudata = _bare_mudata()
    # c2 and c3 are in `t1` under *both* kinds, which `allow_overlap` permits and a single
    # column could not represent.
    mudata.obs["initial_states"] = pd.Categorical(["HSC"] * 2 + ["t1"] * 2 + [None] * 8)
    mudata.obs["terminal_states"] = pd.Categorical([None] * 2 + ["t1"] * 2 + [None] * 2 + ["Ery"] * 3 + [None] * 3)
    mudata.obs["macrostates"] = pd.Categorical(["HSC"] * 2 + ["t1"] * 4 + ["Ery"] * 3 + ["TAC"] * 3)
    return mudata


def _legacy_mudata() -> MuData:
    """An object as a version before this layout wrote it."""
    mudata = _bare_mudata()
    mudata.uns["initial_states"] = {"HSC": ["c0", "c1"]}
    mudata.uns["terminal_states"] = {"Ery": ["c8"], "Mye": ["c9"]}
    mudata.uns["intermediate_states"] = {"TAC": ["c4", "c5"]}
    mudata.uns["fate_state_colors"] = {
        "HSC": "#e41a1c",
        "Ery": "#377eb8",
        "Mye": "#4daf4a",
        "TAC": "#984ea3",
    }
    return mudata


def _palantir_mudata() -> MuData:
    rng = np.random.default_rng(SEED)
    rna = AnnData(rng.random((len(CELLS), len(GENES))))
    act = AnnData(rng.random((len(CELLS), len(ACTIVITY_VAR))))
    rna.obs_names, act.obs_names = CELLS, CELLS
    rna.var_names, act.var_names = GENES, ACTIVITY_VAR

    mudata = MuData({"rna": rna, "activity": act})
    mudata.obs = pd.DataFrame(
        {"cluster": ["red" if i % 2 == 0 else "blue" for i in range(len(CELLS))]},
        index=CELLS,
    )

    rows = np.repeat(np.arange(len(CELLS)), K)
    cols = rng.integers(0, len(CELLS), size=len(CELLS) * K)
    dists = rng.random(len(CELLS) * K)
    mudata.obsp["wnn_distances"] = csr_matrix((dists, (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.obsp["wnn_connectivities"] = csr_matrix((np.ones_like(dists), (rows, cols)), shape=(len(CELLS), len(CELLS)))
    mudata.uns["wnn"] = {"params": {"n_neighbors": K}}
    mudata.obsm["multiscale"] = pd.DataFrame(np.random.default_rng(SEED).random((len(CELLS), 5)), index=CELLS)
    mudata.obsm["eigenvectors"] = pd.DataFrame(np.random.default_rng(SEED).random((len(CELLS), 5)), index=CELLS)
    return mudata


def _run_palantir(cluster_key: str | None) -> MuData:
    extension = PalantirExtension(_palantir_mudata())
    extension.run(
        early_cell=EARLY_CELL,
        knn=10,
        cluster_key=cluster_key,
        num_waypoints=NUM_WAYPOINTS,
        eigvec_key="eigenvectors",
        eigvec_multi_key="multiscale",
    )
    return extension.mudata


def _colour_of(mudata: MuData, kind: str, name: str) -> str:
    """The colour a kind's list gives a state, resolved by name rather than by position."""
    return list(mudata.uns[f"{kind}_colors"])[list(mudata.obs[kind].cat.categories).index(name)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------------------
# the columns
# --------------------------------------------------------------------------------------


def test_a_cell_carries_the_name_of_the_state_it_belongs_to():
    mudata = _recorded_mudata()
    assert mudata.obs["initial_states"].loc["c0"] == "HSC"
    assert mudata.obs["terminal_states"].loc["c6"] == "Ery"


def test_a_cell_in_no_state_of_a_kind_carries_no_value():
    mudata = _recorded_mudata()
    assert pd.isna(mudata.obs["terminal_states"].loc["c0"])
    assert mudata.obs["terminal_states"].notna().sum() == 5


def test_each_kind_records_only_its_own_states():
    mudata = _recorded_mudata()
    assert list(mudata.obs["initial_states"].cat.categories) == ["HSC", "t1"]
    assert list(mudata.obs["terminal_states"].cat.categories) == ["Ery", "t1"]
    assert "TAC" not in list(mudata.obs["initial_states"].cat.categories)


def test_the_kinds_are_independent():
    """A cell in an initial state is not thereby excluded from a terminal one."""
    mudata = _recorded_mudata()
    both = mudata.obs["initial_states"].notna() & mudata.obs["terminal_states"].notna()
    assert both.sum() == 2


# --------------------------------------------------------------------------------------
# one colour per state name — the defect that motivated the change
# --------------------------------------------------------------------------------------


def test_a_name_under_two_kinds_has_one_colour():
    """Assigning per kind gives each list the start of the colour cycle, so an initial and an
    unrelated terminal state come out identical while one state under both comes out twice."""
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)

    assert _colour_of(mudata, "initial_states", "t1") == _colour_of(mudata, "terminal_states", "t1")
    assert _colour_of(mudata, "macrostates", "t1") == _colour_of(mudata, "initial_states", "t1")


def test_distinct_states_have_distinct_colours():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    palette = mudata.uns["atlas_state_palette"]
    assert len(set(palette.values())) == len(palette)


def test_unrelated_states_of_different_kinds_do_not_collide():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    assert _colour_of(mudata, "initial_states", "HSC") != _colour_of(mudata, "terminal_states", "Ery")


def test_every_list_is_aligned_to_its_own_categories():
    """The record is ordered by name and the categories are not, so pairing the two orderings
    would misalign every list while still producing one of the right length."""
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    palette = mudata.uns["atlas_state_palette"]
    for kind in KINDS:
        categories = list(mudata.obs[kind].cat.categories)
        assert list(mudata.uns[f"{kind}_colors"]) == [palette[name] for name in categories]


def test_a_repeated_inference_keeps_the_colours_it_already_gave():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    before = dict(mudata.uns["atlas_state_palette"])

    mudata.obs["terminal_states"] = pd.Categorical([None] * 4 + ["t1"] * 2 + ["Ery"] * 3 + ["new"] * 3)
    migrate_or_assign(mudata)

    after = mudata.uns["atlas_state_palette"]
    assert {k: v for k, v in after.items() if k in before} == before
    assert "new" in after


def test_the_assignment_does_not_depend_on_which_inference_ran():
    """The same names must be coloured the same way however they were produced."""
    one, other = _bare_mudata(), _bare_mudata()
    one.obs["terminal_states"] = pd.Categorical(["Ery"] * 6 + ["Mye"] * 6)
    other.obs["terminal_states"] = pd.Categorical(["Mye"] * 6 + ["Ery"] * 6)
    migrate_or_assign(one)
    migrate_or_assign(other)

    assert one.uns["atlas_state_palette"] == other.uns["atlas_state_palette"]


def test_the_superseded_mapping_agrees_with_the_lists():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    legacy = mudata.uns["fate_state_colors"]
    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert legacy[name] == _colour_of(mudata, kind, name)


# --------------------------------------------------------------------------------------
# restoring what was assigned
# --------------------------------------------------------------------------------------


def test_plotting_overwrites_a_colour_list():
    """The premise of the repair: the lists are the ecosystem's keys, so it writes them."""
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    mudata.obsm["X_umap"] = np.random.default_rng(SEED).random((mudata.n_obs, 2))
    before = _colour_of(mudata, "terminal_states", "t1")

    sc.pl.embedding(mudata, basis="X_umap", color="terminal_states", palette="tab10", show=False)

    assert _colour_of(mudata, "terminal_states", "t1") != before


def test_plotting_leaves_the_record_untouched():
    """What the repair depends on: the record is not named as the convention's keys are."""
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    mudata.obsm["X_umap"] = np.random.default_rng(SEED).random((mudata.n_obs, 2))
    before = dict(mudata.uns["atlas_state_palette"])

    sc.pl.embedding(mudata, basis="X_umap", color="terminal_states", palette="tab10", show=False)

    assert mudata.uns["atlas_state_palette"] == before


def test_restoring_puts_back_exactly_what_was_assigned():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    expected = dict(mudata.uns["atlas_state_palette"])
    mudata.uns["terminal_states_colors"] = ["#000000", "#111111"]

    reset_state_colors(mudata)

    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert _colour_of(mudata, kind, name) == expected[name]


def test_restoring_recovers_every_list_at_once():
    """A reconciliation between the lists could not recover this: nothing would be left to
    reconcile from."""
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    expected = dict(mudata.uns["atlas_state_palette"])
    for kind in KINDS:
        mudata.uns[f"{kind}_colors"] = ["#000000"] * len(mudata.obs[kind].cat.categories)

    reset_state_colors(mudata)

    for kind in KINDS:
        for name in mudata.obs[kind].cat.categories:
            assert _colour_of(mudata, kind, name) == expected[name]


def test_restoring_without_a_record_says_so():
    mudata = _recorded_mudata()
    with pytest.raises(KeyError, match="atlas_state_palette"):
        reset_state_colors(mudata)


# --------------------------------------------------------------------------------------
# the guard protecting the superseded mapping
# --------------------------------------------------------------------------------------


def test_a_column_that_would_consume_the_superseded_mapping_is_refused():
    """`uns["fate_state_colors"]` is a mapping under a name the convention reads positionally.
    This is the test that fails for whoever later adds an argmax column called `fate_state`."""
    mudata = _recorded_mudata()
    mudata.obs["fate_state"] = pd.Categorical(["a"] * 6 + ["b"] * 6)

    with pytest.raises(ValueError, match="fate_state"):
        migrate_or_assign(mudata)


def test_without_such_a_column_colours_are_written():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    assert mudata.uns["atlas_state_palette"]


# --------------------------------------------------------------------------------------
# naming a state does not change what was inferred
# --------------------------------------------------------------------------------------


def test_naming_states_after_a_cluster_only_renames_them():
    """`cluster_key` is documented as renaming inferred states. It must not also merge them,
    which would let a naming choice change how many fates are reported."""
    bare, named = _run_palantir(None), _run_palantir("cluster")

    assert bare.obsm["fate_probabilities"].shape == named.obsm["fate_probabilities"].shape
    np.testing.assert_allclose(
        bare.obsm["fate_probabilities"].to_numpy(),
        named.obsm["fate_probabilities"].to_numpy(),
    )
    assert list(bare.obsm["fate_probabilities"].columns) != list(named.obsm["fate_probabilities"].columns)


def test_states_claiming_one_cluster_are_disambiguated():
    named = _run_palantir("cluster")
    names = list(named.uns["initial_states"]) + list(named.uns["terminal_states"])

    assert len(set(names)) == len(names)
    for name in names:
        assert re.sub(r"_\d+$", "", name) in {"red", "blue"}


def test_disambiguation_is_repeatable():
    assert list(_run_palantir("cluster").uns["terminal_states"]) == list(
        _run_palantir("cluster").uns["terminal_states"]
    )


def test_a_cluster_no_state_corresponds_to_is_not_a_category():
    """The annotation states are named after is a naming source, not a category set."""
    named = _run_palantir("cluster")
    for kind in KINDS:
        for name in named.obs[kind].cat.categories:
            assert name in set(named.uns["initial_states"]) | set(named.uns["terminal_states"])


def test_every_fate_probability_column_carries_a_colour():
    """`atlas.pl` indexes the colour mapping by fate-probability column name, so a name applied
    to one side and not the other raises there. This change does not otherwise touch it."""
    for cluster_key in (None, "cluster"):
        mudata = _run_palantir(cluster_key)
        for column in mudata.obsm["fate_probabilities"].columns:
            assert column in mudata.uns["fate_state_colors"]
            assert column in set(mudata.obs["terminal_states"].cat.categories)


# --------------------------------------------------------------------------------------
# the coarse states, and what is derived from them
# --------------------------------------------------------------------------------------


def test_the_coarse_states_are_recorded_whatever_produced_the_object():
    for mudata in (_run_palantir(None), _run_palantir("cluster")):
        assert "macrostates" in mudata.obs
        assert mudata.obs["macrostates"].notna().any()


def test_nothing_lies_between_states_where_the_coarse_states_are_their_union():
    assert _intermediate_states(_run_palantir("cluster")) == {}


def test_intermediate_states_are_found_where_a_coarse_graining_exists():
    """Derived cell-wise, as the superseded `uns["intermediate_states"]` always was.

    A coarse state is designated initial or terminal through a subset of its cells — CellRank
    marks only the most representative — so its remaining cells are neither, and are reported
    here. `t1` appears despite being both an initial and a terminal state because only `c2` and
    `c3` carry those assignments, leaving `c4` and `c5`. This is what the previous
    implementation did, and preserving it is what keeps the deprecated key unchanged.
    """
    mudata = _recorded_mudata()
    assert set(_intermediate_states(mudata)) == {"TAC", "t1"}
    assert _intermediate_states(mudata)["TAC"] == ["c9", "c10", "c11"]


def test_deriving_without_coarse_states_says_so():
    mudata = _bare_mudata()
    mudata.obs["terminal_states"] = pd.Categorical(["Ery"] * 12)
    with pytest.raises(KeyError, match="macrostates"):
        _intermediate_states(mudata)


def test_intermediate_states_are_not_a_kind_of_their_own():
    mudata = _recorded_mudata()
    migrate_or_assign(mudata)
    assert "intermediate_states" not in mudata.obs
    assert "intermediate_states_colors" not in mudata.uns


# --------------------------------------------------------------------------------------
# converting an object written by an earlier version
# --------------------------------------------------------------------------------------


def test_migrating_records_the_states_it_finds():
    mudata = _legacy_mudata()
    migrate_states(mudata)

    assert list(mudata.obs["initial_states"].cat.categories) == ["HSC"]
    assert set(mudata.obs["terminal_states"].cat.categories) == {"Ery", "Mye"}
    assert mudata.obs["initial_states"].loc["c0"] == "HSC"


def test_migrating_reconstructs_the_coarse_states_as_the_union():
    mudata = _legacy_mudata()
    migrate_states(mudata)
    assert set(mudata.obs["macrostates"].cat.categories) == {"HSC", "Ery", "Mye", "TAC"}
    assert mudata.obs["macrostates"].notna().sum() == 6  # c0 c1 | c8 c9 | c4 c5


def test_migrating_carries_the_colours_over():
    mudata = _legacy_mudata()
    migrate_states(mudata)
    assert mudata.uns["atlas_state_palette"]["HSC"] == "#e41a1c"
    assert _colour_of(mudata, "terminal_states", "Ery") == "#377eb8"


def test_migrating_leaves_what_was_there():
    mudata = _legacy_mudata()
    migrate_states(mudata)
    for key in ("initial_states", "terminal_states", "intermediate_states", "fate_state_colors"):
        assert key in mudata.uns


def test_migrating_twice_changes_nothing():
    mudata = _legacy_mudata()
    migrate_states(mudata)
    before = list(mudata.obs["macrostates"].astype(object)), dict(mudata.uns["atlas_state_palette"])

    migrate_states(mudata)

    assert (list(mudata.obs["macrostates"].astype(object)), dict(mudata.uns["atlas_state_palette"])) == before


def test_migrating_an_object_with_nothing_to_convert_says_so():
    with pytest.raises(KeyError, match="nothing to convert"):
        migrate_states(_bare_mudata())


def test_a_migrated_object_agrees_with_a_freshly_computed_one():
    """Migration is exact: the union it reconstructs is what a fresh run writes."""
    fresh = _run_palantir("cluster")
    legacy = _bare_mudata(len(CELLS))
    legacy.obs_names = CELLS
    legacy.uns["initial_states"] = {k: list(v) for k, v in fresh.uns["initial_states"].items()}
    legacy.uns["terminal_states"] = {k: list(v) for k, v in fresh.uns["terminal_states"].items()}
    legacy.uns["intermediate_states"] = {}

    migrate_states(legacy)

    assert set(legacy.obs["macrostates"].cat.categories) == set(fresh.obs["macrostates"].cat.categories)
    assert list(legacy.obs["macrostates"].astype(object)) == list(fresh.obs["macrostates"].astype(object))


# --------------------------------------------------------------------------------------
# the superseded layout keeps working
# --------------------------------------------------------------------------------------


def test_both_layouts_are_written():
    mudata = _run_palantir("cluster")
    assert "initial_states" in mudata.uns and "initial_states" in mudata.obs
    assert "terminal_states" in mudata.uns and "terminal_states" in mudata.obs
    assert isinstance(mudata.uns["fate_state_colors"], dict)


def test_the_superseded_dictionaries_still_map_names_to_cells():
    mudata = _run_palantir("cluster")
    for key in ("initial_states", "terminal_states"):
        for name, cells in mudata.uns[key].items():
            assert isinstance(name, str)
            assert all(cell in set(mudata.obs_names) for cell in cells)


def migrate_or_assign(mudata: MuData) -> None:
    """Assign colours for states recorded directly, without running an inference."""
    from atlas.tl.utils import _assign_state_colors

    _assign_state_colors(mudata)


# --------------------------------------------------------------------------------------
# a cell belonging to states of two kinds
# --------------------------------------------------------------------------------------


def test_the_coarse_states_take_the_initial_assignment_and_warn():
    """`macrostates` admits one value per cell, so an overlap has to resolve. Initial takes
    precedence, following CellRank, and the caller is told which cells were affected."""
    initial = _states_to_column({"HSC": ["c0", "c1"]}, _bare_mudata().obs_names)
    terminal = _states_to_column({"Ery": ["c1"], "Mye": ["c8"]}, _bare_mudata().obs_names)

    with pytest.warns(UserWarning, match="both an initial and a terminal state"):
        macrostates = _resolve_overlap(initial, terminal)

    assert macrostates.loc["c1"] == "HSC"
    assert macrostates.loc["c8"] == "Mye"
    assert set(macrostates.cat.categories) == {"HSC", "Ery", "Mye"}


def test_no_warning_where_nothing_overlaps():
    initial = _states_to_column({"HSC": ["c0"]}, _bare_mudata().obs_names)
    terminal = _states_to_column({"Ery": ["c8"]}, _bare_mudata().obs_names)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _resolve_overlap(initial, terminal)


def test_both_assignments_survive_the_overlap():
    """Resolving the single-valued view must not discard either kind's record of the cell."""
    mudata = _recorded_mudata()
    assert mudata.obs["initial_states"].loc["c2"] == "t1"
    assert mudata.obs["terminal_states"].loc["c2"] == "t1"
