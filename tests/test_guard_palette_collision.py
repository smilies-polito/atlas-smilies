"""The guard protecting the superseded colour mapping.

Covers :func:`atlas.tl.utils._guard_palette_collision`. See
``test_assign_state_colors.py`` for the property these tests take part in.
"""

import pandas as pd
import pytest

from atlas.tl.utils import _assign_state_colors


def test_a_column_that_would_consume_the_superseded_mapping_is_refused(recorded_mudata):
    """`uns["fate_state_colors"]` is a mapping under a name the convention reads positionally.
    This is the test that fails for whoever later adds an argmax column called `fate_state`."""
    mudata = recorded_mudata()
    mudata.obs["fate_state"] = pd.Categorical(["a"] * 6 + ["b"] * 6)

    with pytest.raises(ValueError, match="fate_state"):
        _assign_state_colors(mudata)


def test_without_such_a_column_colours_are_written(recorded_mudata):
    mudata = recorded_mudata()
    _assign_state_colors(mudata)
    assert mudata.uns["atlas_state_palette"]
