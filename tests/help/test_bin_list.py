"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_bin_list():
    """Test that bin_list returns expected output"""

    lenl = 100
    steps = 8
    np.random.seed(0)

    lst = np.random.rand(lenl)
    lst[5:10] = np.nan
    bins = hlppy.help.bin_list(lst, steps=steps)

    assert isinstance(bins, np.ndarray)
    assert len(bins) == lenl
    assert all(bins[5:10] == np.array(['0']*5))
    assert all(np.unique(bins) == np.arange(steps+1).astype(np.str))
