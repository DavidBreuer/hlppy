"""Test"""

import hlppy
import hlppy.help


def test_sort_natural():

    lst = ['imp_10', 'imp_100', 'imp_101', 'imp_12']
    srt = hlppy.help.sort_natural(lst)

    # preserves elements
    assert set(srt) == set(lst)

    # fixes order
    nms = [int(ele[4:]) for ele in srt]
    assert nms == sorted(nms)
