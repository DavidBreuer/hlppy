"""Test"""

import hlppy
import hlppy.help


def test_flatten_list():
    """Test that flatten_list returns expected output"""

    lst = [[True, False], ['foo'], [], [1.0, 4.2]]
    assert hlppy.help.flatten_list(lst) == [True, False, 'foo', 1.0, 4.2]

    lst_empty = []
    assert hlppy.help.flatten_list(lst_empty) == lst_empty
