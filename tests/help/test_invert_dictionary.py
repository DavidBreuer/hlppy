"""Test"""

import hlppy
import hlppy.help


def test_invert_dictionary():

    dct = dict()
    dct[0] = 'A'
    dct[1] = 'b'

    cpy = hlppy.help.invert_dictionary(dct)

    # correct type
    assert isinstance(cpy, dict)
    # correct keys
    assert list(cpy.keys()) == list(dct.values())
    # correct values
    assert list(cpy.values()) == list(dct.keys())
