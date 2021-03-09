"""Test"""

import hlppy
import hlppy.help


def test_append_dictionary():

    dct = dict()
    dct[0] = 'A'
    dct[1] = 'b'

    # append values to list
    cpy = hlppy.help.append_dictionary(dct, dct)

    # correct type
    assert isinstance(cpy, dict)
    # correct keys
    assert list(cpy.keys()) == list(dct.keys())
    # correct values
    assert all([isinstance(val, list) for val in cpy.values()])

    # just set values
    cpy = hlppy.help.append_dictionary(dct, dct, listed=False)

    # check identity
    assert cpy == dct
