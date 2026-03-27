import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from atlas.tl.evaluate import _hard_ai


def test_hard_ai():
    D = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], index=["c1", "c2", "c3"], columns=["c1", "c2", "c3"]
    )

    f = pd.DataFrame({"A": [0.9, 0.8, 0.1], "B": [0.1, 0.2, 0.9]}, index=["c1", "c2", "c3"])

    expected = pd.Series({"c1": 1.0, "c2": 1.0, "c3": np.nan})
    assert_series_equal(_hard_ai(f, D), expected)
