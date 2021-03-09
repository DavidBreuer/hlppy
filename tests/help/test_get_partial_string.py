"""Test"""

import hlppy
import hlppy.help


def test_get_partial_string():

    string = 'aaa-123-xxx'

    assert hlppy.help.get_partial_string(string) == string
    assert hlppy.help.get_partial_string(string, start=1) == string[4:]
    assert hlppy.help.get_partial_string(string, stop=-1) == string[:-4]
    assert hlppy.help.get_partial_string(string, start=1, stop=2) == string[4:-4]
    assert hlppy.help.get_partial_string(string, sep='.') == string
