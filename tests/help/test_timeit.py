"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_timeit():

    def dummy_func():
        return np.sum(np.arange(10000))

    reps = 7
    result, durations = hlppy.help.timeit(dummy_func, reps=reps)

    assert result == 49995000
    assert len(durations) == reps
    assert (np.array(durations) > 0).all()

    del dummy_func
