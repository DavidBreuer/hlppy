"""Test"""

import pandas as pd

import hlppy
import hlppy.help


def test_count_frequencies():

    ara = [0, 1, 1, 1, 4]
    arb = ['a', 'c', 'c', 'd', 'd']
    tab = pd.DataFrame([], index=range(len(ara)))
    tab['A'] = ara
    tab['B'] = arb

    levz, freqz = hlppy.help.count_frequencies(tab)

    assert isinstance(levz, list)
    assert isinstance(freqz, list)

    assert len(levz) == 2
    assert len(freqz) == 2

    assert list(levz[0]) == [0, 1, 4]
    assert list(levz[1]) == ['a', 'c', 'd']
    assert list(freqz[0]) == [1, 3, 1]
    assert list(freqz[1]) == [1, 2, 2]
