"""Test"""

import hlppy
import hlppy.help


def test_stdout_to_string():
    """Test that stdout_to_strings returns expected output"""

    def dummy_func(vara, varb, varc, vard=10, vare=100):
        """dummy function for testing"""
        print(vara, varb, varc, vard, vare)

    assert '42' in hlppy.help.stdout_to_string(dummy_func, 1, 2, 42)
    assert '12' in hlppy.help.stdout_to_string(dummy_func, 1, 2, [1, 2], 12)
    assert 'foo' in hlppy.help.stdout_to_string(dummy_func, 'foo', True, -3)
    assert '4' in hlppy.help.stdout_to_string(dummy_func, 1, 2, 3, vard=4)

    del dummy_func
