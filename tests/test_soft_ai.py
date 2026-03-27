import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from atlas.tl.evaluate import _soft_ai


def test_soft_ai():
    D = pd.DataFrame(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]], index=["c1", "c2", "c3"], columns=["c1", "c2", "c3"]
    )

    f = pd.DataFrame({"A": [1, 1, 0], "B": [0, 0, 1]}, index=["c1", "c2", "c3"])

    expected = pd.Series({"c1": 2.0, "c2": 2.0, "c3": np.nan})
    assert_series_equal(_soft_ai(f, D), expected)
