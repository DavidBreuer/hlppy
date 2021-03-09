"""Test"""

import hlppy
import hlppy.help


def test_sort_dictionary():
    """Test that sort_dictionary returns expected output"""

    dct = dict(zip([1, 3, 2], [9, 8, 7]))

    cpy = hlppy.help.sort_dictionary(dct, which='key')

    assert isinstance(cpy, dict)
    assert list(cpy.keys()) == [1, 2, 3]

    cpy = hlppy.help.sort_dictionary(dct, which='value')

    assert isinstance(cpy, dict)
    assert list(cpy.values()) == [7, 8, 9]
