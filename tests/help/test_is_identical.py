"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_is_identical():

    assert hlppy.help.is_identical([0, 0, 0])
    assert hlppy.help.is_identical(['0', '0', '0'])
    assert hlppy.help.is_identical([np.nan, np.nan, np.nan])

    assert not hlppy.help.is_identical([0, 0, 1])
    assert not hlppy.help.is_identical([0, '0', np.nan])
