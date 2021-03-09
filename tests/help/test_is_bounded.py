"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_is_bounded():

    arr = [0, 1, 4]

    exp = np.full(len(arr), False)
    exp[1] = True

    assert np.all(hlppy.help.is_bounded(arr, 1, 3) == exp)

    assert np.all(hlppy.help.is_bounded(np.array(arr), 1, 3) == exp)
