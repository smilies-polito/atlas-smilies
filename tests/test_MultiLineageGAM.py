import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from muon import MuData

from atlas.pl.utils import MultiBranchGAM
from atlas.tl import MultiLineageGAM

SEED = 42
N_CELLS = 200
CELLS = [f"cell{i}" for i in range(N_CELLS)]
TFS = ["GATA1", "SPI1", "KLF4"]
GENES = ["KLF1", "HBB", "SLC4A1"]
LINEAGES = ["Ery", "Mye"]


def _mudata(fate: pd.DataFrame | None = None) -> MuData:
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 1, N_CELLS)

    rna = AnnData(rng.random((N_CELLS, len(TFS))).astype(np.float32))
    rna.obs_names, rna.var_names = CELLS, TFS
    activity = AnnData(rng.random((N_CELLS, len(GENES))).astype(np.float32))
    activity.obs_names, activity.var_names = CELLS, GENES

    mudata = MuData({"rna": rna, "activity": activity})
    mudata.obs["pseudotime"] = time
    if fate is None:
        fate = pd.DataFrame(np.c_[time, 1 - time], index=CELLS, columns=LINEAGES)
    mudata.obsm["fate_probabilities"] = fate
    return mudata


def _fitted(mudata: MuData, genes, n_points: int = 50) -> MultiLineageGAM:
    model = MultiLineageGAM(mudata=mudata, ptf="GATA1", genes=genes)
    model.fit()
    model.predict(n_points=n_points)
    return model


def test_the_curves_match_the_superseded_procedure_exactly():
    """The guarantee the change rests on: what is computed does not move."""
    mudata = _mudata()

    old = MultiBranchGAM(mudata=mudata, ptf="GATA1", gene="KLF1")
    old.fit()
    old.predict(n_points=50)
    new = _fitted(mudata, "KLF1")

    assert set(new.predictions) == set(old.predictions)
    for lineage, before in old.predictions.items():
        after = new.predictions[lineage]
        assert np.array_equal(after["t_grid"], before["t_grid"])
        for key in ("gex", "gex_lower", "gex_upper"):
            assert np.array_equal(after[key], before[key])
        curve = after["genes"]["KLF1"]
        assert np.array_equal(curve["act"], before["act"])
        assert np.array_equal(curve["act_lower"], before["act_lower"])
        assert np.array_equal(curve["act_upper"], before["act_upper"])


def test_the_factor_is_the_same_curve_however_many_genes_are_fitted():
    mudata = _mudata()
    one, several = _fitted(mudata, "KLF1"), _fitted(mudata, GENES)

    for lineage, alone in one.predictions.items():
        together = several.predictions[lineage]
        for key in ("t_grid", "gex", "gex_lower", "gex_upper"):
            assert np.array_equal(alone[key], together[key])


def test_a_gene_is_the_same_curve_however_many_are_fitted():
    mudata = _mudata()
    alone, together = _fitted(mudata, "HBB"), _fitted(mudata, GENES)

    for lineage in alone.predictions:
        one = alone.predictions[lineage]["genes"]["HBB"]
        many = together.predictions[lineage]["genes"]["HBB"]
        for key in ("act", "act_lower", "act_upper"):
            assert np.array_equal(one[key], many[key])


def test_the_grid_belongs_to_the_lineage_not_to_a_gene():
    model = _fitted(_mudata(), GENES)
    for prediction in model.predictions.values():
        assert "t_grid" in prediction
        assert all("t_grid" not in curve for curve in prediction["genes"].values())


def test_every_gene_has_an_entry_under_its_lineage():
    model = _fitted(_mudata(), GENES)
    for prediction in model.predictions.values():
        assert sorted(prediction["genes"]) == sorted(GENES)
        for curve in prediction["genes"].values():
            assert sorted(curve) == ["act", "act_lower", "act_upper"]


def test_the_models_are_grouped_by_lineage():
    model = _fitted(_mudata(), GENES)
    assert sorted(model.models) == sorted(LINEAGES)
    for fitted in model.models.values():
        assert sorted(fitted) == ["gam_exp", "gams_act", "weights"]
        assert sorted(fitted["gams_act"]) == sorted(GENES)


def test_one_gene_may_be_given_as_a_name():
    model = _fitted(_mudata(), "KLF1")
    for prediction in model.predictions.values():
        assert list(prediction["genes"]) == ["KLF1"]


def test_a_lineage_with_negligible_weight_is_skipped_with_a_warning():
    time = np.linspace(0, 1, N_CELLS)
    fate = pd.DataFrame(np.c_[time, np.zeros(N_CELLS)], index=CELLS, columns=LINEAGES)

    with pytest.warns(UserWarning, match="no cells transitioning towards Mye"):
        model = _fitted(_mudata(fate), "KLF1")

    assert list(model.predictions) == ["Ery"]


def test_a_lineage_that_can_be_fitted_still_is():
    time = np.linspace(0, 1, N_CELLS)
    fate = pd.DataFrame(np.c_[time, np.zeros(N_CELLS)], index=CELLS, columns=LINEAGES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _fitted(_mudata(fate), "KLF1")

    assert model.predictions["Ery"]["gex"].shape == (50,)


@pytest.mark.parametrize(
    ("ptf", "genes", "missing"),
    [("nope", "KLF1", "nope"), ("GATA1", "nope", "nope"), ("GATA1", ["KLF1", "nope"], "nope")],
)
def test_naming_something_absent_says_what_was_not_found(ptf, genes, missing):
    with pytest.raises(KeyError, match=missing):
        MultiLineageGAM(mudata=_mudata(), ptf=ptf, genes=genes)


def test_an_absent_key_is_reported():
    mudata = _mudata()
    with pytest.raises(KeyError, match="nope"):
        MultiLineageGAM(mudata=mudata, ptf="GATA1", genes="KLF1", time_key="nope")
    with pytest.raises(KeyError, match="nope"):
        MultiLineageGAM(mudata=mudata, ptf="GATA1", genes="KLF1", fate_prob_key="nope")


@pytest.mark.parametrize("present", ["rna", "activity"])
def test_a_missing_modality_is_named(present):
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 1, N_CELLS)

    only = AnnData(rng.random((N_CELLS, 3)).astype(np.float32))
    only.obs_names = CELLS
    only.var_names = TFS if present == "rna" else GENES

    mudata = MuData({present: only})
    mudata.obs["pseudotime"] = time
    mudata.obsm["fate_probabilities"] = pd.DataFrame(np.c_[time, 1 - time], index=CELLS, columns=LINEAGES)

    absent = "activity" if present == "rna" else "rna"
    with pytest.raises(KeyError, match=absent):
        MultiLineageGAM(mudata=mudata, ptf="GATA1", genes="KLF1")


def test_both_modalities_present_fits():
    model = _fitted(_mudata(), "KLF1")
    assert sorted(model.models) == sorted(LINEAGES)
