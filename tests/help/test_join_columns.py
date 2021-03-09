"""Test"""

import pandas as pd

import hlppy
import hlppy.help


def test_join_columns(y_fake):

    sep = '.'
    num = 3
    cols = y_fake.columns[:num]

    out = hlppy.help.join_columns(y_fake, cols, sep=sep)

    # correct type
    assert isinstance(out, pd.Series)
    # correct length
    assert len(out) == len(y_fake)
    # correct number of separators
    assert set([len(val.split(sep)) for val in out]) == set([num])
