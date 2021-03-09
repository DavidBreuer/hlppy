"""Test"""

import hlppy
import hlppy.help


def test_example():
    assert hlppy.help.example(2) == 4
