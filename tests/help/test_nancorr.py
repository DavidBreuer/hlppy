"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_nancorr():

    mat = np.random.rand(100, 10)
    idi = 0
    vec = mat[idi]

    out = hlppy.help.nancorr(mat, vec)

    one = 1 + 1e-12

    assert isinstance(out, np.ndarray)
    assert len(out) == len(mat)
    assert np.allclose(out[idi], 1.0)
    assert (np.max(out) <= one) and (np.max(out) >= -one)
