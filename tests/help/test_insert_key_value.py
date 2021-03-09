"""Test"""

import hlppy
import hlppy.help


def test_insert_key_value():

    dct = dict()
    dct['bla'] = 0
    dct['blu'] = 1

    new = hlppy.help.insert_key_value(dct, 0, 'key', 'val')
    assert list(new.keys())[0] == 'key'
    assert len(new.keys()) == len(dct.keys()) + 1

    new = hlppy.help.insert_key_value(dct, -1, 'key', 'val')
    assert list(new.keys())[2] == 'key'

    new = hlppy.help.insert_key_value(dct, 'bla', 'key', 'val')
    assert list(new.keys())[1] == 'key'

    new = hlppy.help.insert_key_value(dct, 'blx', 'bla', 'val')
    assert len(new.keys()) == len(dct.keys())
