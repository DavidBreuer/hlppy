"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_is_intersecting():
    """Test that is_intersecting returns expected output"""

    veca = [1, 2, 3]

    assert hlppy.help.is_intersecting(veca, [1, 5])
    assert hlppy.help.is_intersecting(veca, veca)
    assert hlppy.help.is_intersecting(veca, np.array(veca))
    assert hlppy.help.is_intersecting(veca, set(veca))

    assert not hlppy.help.is_intersecting([], [])
    assert not hlppy.help.is_intersecting(veca, [])
    assert not hlppy.help.is_intersecting(veca, [4, 5])
