"""Test"""

import hlppy
import hlppy.help


def test_make_name():
    """Test that make_name returns expected output"""

    dct = dict([('a', 1), ('b', 'b')])
    assert hlppy.help.make_name(dct) == 'a=1-b=b'
    assert hlppy.help.make_name(dct, exclude=['a']) == 'b=b'
    assert hlppy.help.make_name(dct, include=['b']) == 'b=b'
