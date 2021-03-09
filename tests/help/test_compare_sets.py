"""Test"""

import hlppy
import hlppy.help


def test_compare_sets():
    """Test that compare_sets returns expected output"""

    seta = [1, 2, 3]
    setb = [3, 4, 5, 6]

    outs = hlppy.help.compare_sets(seta, setb)
    union, intersec, symdiff, diffab, diffba = outs

    assert isinstance(outs, tuple)
    for out in outs:
        assert isinstance(out, set)
    assert union == set(seta + setb)
    assert intersec == set([3])
    assert symdiff == diffab.union(diffba)

    setb = list(range(100))
    outs = hlppy.help.compare_sets(seta, setb)
    assert isinstance(outs, tuple)
