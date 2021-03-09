"""Test"""

import time

import numpy as np

import hlppy
import hlppy.help


def test_benchmark(capsys):

    def dummy_func_fast():
        return np.sum(np.arange(10000))

    def dummy_func_slow():
        time.sleep(1e-3)
        return np.sum(np.arange(10000))

    reps = 6
    names, results, durations = hlppy.help.benchmark(dummy_func_fast,
                                                       dummy_func_slow,
                                                       reps=reps)

    captured = capsys.readouterr()

    # correct shapes
    assert len(results) == 2
    assert len(durations) == 2
    assert len(durations[0]) == reps

    # correct values
    assert (np.array(durations[0]) > 0).all()
    assert np.nanmean(durations[0]) <= np.nanmean(durations[1])

    # correct prints
    assert dummy_func_fast.__name__ in captured.out
    assert dummy_func_slow.__name__ in captured.out

    del dummy_func_fast
    del dummy_func_slow
