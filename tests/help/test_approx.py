"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_approx():

    values = np.linspace(0, 100, 20)
    targets = [20, 90, 50]
    out = hlppy.help.approx(np.atleast_2d(targets).T,
                               np.atleast_2d(values).T)

    assert isinstance(out, np.ndarray)
    assert len(out) == len(values)
    assert len(set(out)) == len(targets)
