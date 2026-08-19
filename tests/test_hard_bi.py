import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from atlas.tl.evaluate import _hard_bi


def test_hard_bi():
    D = pd.DataFrame([[0, 2, 3], [2, 0, 1], [3, 1, 0]], index=["c1", "c2", "c3"], columns=["c1", "c2", "c3"])

    f = pd.DataFrame(np.eye(3), index=["c1", "c2", "c3"])

    expected = pd.Series([2.0, 1.0, 1.0], index=["c1", "c2", "c3"])
    assert_series_equal(_hard_bi(f, D), expected)


def test_a_terminal_state_no_cell_is_assigned_to_is_skipped():
    """A recorded terminal state that never wins contributes no candidate distance.

    Every other case here assigns at least one cell to every column, so the skip had
    never been taken and a column no cell claims would have raised on an empty mean.
    """
    cells = ["c1", "c2", "c3"]
    D = pd.DataFrame([[0, 2, 3], [2, 0, 1], [3, 1, 0]], index=cells, columns=cells)

    # "C" is recorded but never the strongest, so no cell is assigned to it.
    f = pd.DataFrame(
        [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]],
        index=cells,
        columns=["A", "B", "C"],
    )

    # c1 -> A, c2 -> B, c3 -> A; distances to the one other populated state only.
    expected = pd.Series([2.0, 1.5, 1.0], index=cells)
    assert_series_equal(_hard_bi(f, D), expected)
