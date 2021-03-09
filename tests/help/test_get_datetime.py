"""Test"""


import datetime

import hlppy
import hlppy.help


def test_get_datetime():

    out = hlppy.help.get_datetime(None)

    assert isinstance(out, str)
    assert out[8:9] == '-'
    assert (out[:8] + out[9:]).isdecimal()

    inp = datetime.datetime.now()
    out = hlppy.help.get_datetime(inp)

    assert isinstance(out, str)
    assert out[8:9] == '-'
    assert (out[:8] + out[9:]).isdecimal()
