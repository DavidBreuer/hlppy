"""Test"""

import hlppy
import hlppy.help


def test_is_unique():

    assert hlppy.help.is_unique([0, 1, 2, 'a', 'B'])

    assert not hlppy.help.is_unique([0, 1, 2, 'a', 'B', 1])
