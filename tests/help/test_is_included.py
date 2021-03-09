"""Test"""

import hlppy
import hlppy.help


def test_is_included():
    """Test that is_included returns expected output"""

    strngs = ['a', 'b', 'xyz']
    lsts = ['AAAa', 'xxx', 'abc', 'abc123xyz']

    incs = hlppy.help.is_included(strngs, lsts, checkmode=any)
    assert incs == [True, False, True, True]

    incs = hlppy.help.is_included(strngs, lsts, checkmode=all)
    assert incs == [False, False, False, True]

    incs = hlppy.help.is_included('xxx', lsts, checkmode=any)
    assert incs == [False, True, False, False]
