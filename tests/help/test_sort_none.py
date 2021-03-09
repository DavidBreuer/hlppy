"""Test"""

import hlppy
import hlppy.help


def test_sort_none():
    """Test that sorting of lists with None etc. works"""

    vec = [None, 'b', None, 'X', 'a']
    exp = ['None', 'None', 'X', 'a', 'b']

    new, order = hlppy.help.sort_none(vec)

    assert new == exp
    assert len(order) == len(exp)

    vec = [None, 'b', None, (1, 2, 3), 'a']
    exp = ['(1, 2, 3)', 'None', 'None', 'a', 'b']

    new, order = hlppy.help.sort_none(vec)

    assert new == exp
    assert len(order) == len(exp)
