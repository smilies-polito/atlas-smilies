import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from atlas.tl.evaluate import _hard_bi


def test_hard_bi():
    D = pd.DataFrame([[0, 2, 3], [2, 0, 1], [3, 1, 0]], index=["c1", "c2", "c3"], columns=["c1", "c2", "c3"])

    f = pd.DataFrame(np.eye(3), index=["c1", "c2", "c3"])

    expected = pd.Series([2.0, 1.0, 1.0], index=["c1", "c2", "c3"])
    assert_series_equal(_hard_bi(f, D), expected)
