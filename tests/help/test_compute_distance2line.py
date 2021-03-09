"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_compute_distance2line():
    """Test that compute_distance2line returns expected output"""

    pst = np.array([0, 0])
    psp = np.array([1, 1])
    ppp = np.array([[1, 0],
                    [0, 1],
                    [2, 0],
                    [0, 10]])

    dist, line = hlppy.help.compute_distance2line(ppp, pst, psp)

    exp = np.array([-0.70710678,  0.70710678, -1.41421356,  7.07106781])

    assert isinstance(dist, np.ndarray)
    assert len(dist) == len(ppp)
    assert np.allclose(dist, exp)

    assert isinstance(line, np.ndarray)
    assert line.shape == (2, 2)
