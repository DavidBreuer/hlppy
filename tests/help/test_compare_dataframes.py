"""Test"""

import numpy as np
import pandas as pd

import hlppy
import hlppy.help


def test_compare_dataframes():
    """Test that compare_dataframes returns expected output"""

    arr = np.reshape(np.arange(50), (5, -1))

    taba = pd.DataFrame(arr)
    tabb = pd.DataFrame(arr)

    # works with arrays
    eql = hlppy.help.compare_dataframes(arr, arr)
    assert eql

    # works with shape mismatch
    eql = hlppy.help.compare_dataframes(arr[:3], arr)
    assert not eql

    # works with data frames
    eql = hlppy.help.compare_dataframes(taba, tabb)
    assert eql

    tabb.iloc[1:2, 5:8] = np.nan
    tabb.iloc[3, 2:4] = -1

    eql = hlppy.help.compare_dataframes(taba, tabb)
    assert not eql
