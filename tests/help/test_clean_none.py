"""Test"""

import hlppy
import hlppy.help


def test_clean_none():
    """Test that cleaning of None elements etc. works"""

    assert hlppy.help.clean_none(None) == 'None'
    assert hlppy.help.clean_none(1) == 1
    assert hlppy.help.clean_none((9, 1, 1)) == '(9, 1, 1)'
