"""Storing a neighbour graph on the object being annotated.

Covers :func:`atlas.pp._preprocessing._write_graph`, the single place the graph contract
is written. Both producing routes end here, so its guards are what stop a graph and the
object it is stored on from drifting apart.
"""

import numpy as np
import pytest
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

from atlas.pp._preprocessing import _write_graph

CELLS = [f"cell{i}" for i in range(6)]


def _object(names=CELLS) -> MuData:
    rna = AnnData(np.zeros((len(names), 2), dtype=np.float32))
    rna.obs_names = list(names)
    return MuData({"rna": rna})


def _with_graph(names=CELLS, key: str = "wnn") -> MuData:
    mudata = _object(names)
    n = len(names)
    matrix = csr_matrix(np.eye(n, dtype=np.float32))
    mudata.obsp["wnn_distances"] = matrix
    mudata.obsp["wnn_connectivities"] = matrix
    mudata.uns[key] = {"distances_key": "wnn_distances", "connectivities_key": "wnn_connectivities"}
    return mudata


def test_a_graph_is_stored_with_its_provenance():
    target, source = _object(), _with_graph()

    _write_graph(target, source, key="wnn", route="wnn", n_neighbors=5)

    assert target.uns["wnn"]["atlas"] == {"route": "wnn", "n_neighbors": 5}
    assert "wnn_distances" in target.obsp and "wnn_connectivities" in target.obsp


def test_a_source_carrying_no_record_under_the_key_is_named():
    target, source = _object(), _object()

    with pytest.raises(KeyError, match="wnn not in the computed object's .uns"):
        _write_graph(target, source, key="wnn", route="wnn")


def test_observations_that_do_not_agree_are_refused_rather_than_misaligned():
    """Storing a graph computed on other cells would silently misalign its indices."""
    target = _object([f"other{i}" for i in range(6)])
    source = _with_graph()

    with pytest.raises(ValueError, match="would misalign the graph"):
        _write_graph(target, source, key="wnn", route="wnn")


def test_the_same_observations_in_a_different_order_are_refused():
    target = _object(list(reversed(CELLS)))
    source = _with_graph()

    with pytest.raises(ValueError, match="in a different order or number"):
        _write_graph(target, source, key="wnn", route="wnn")
