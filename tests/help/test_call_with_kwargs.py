"""Test"""

import hlppy
import hlppy.help


def test_call_with_kwargs():
    """Test that call_with_kwargs returns expected output"""

    # define dummy function
    def dummy_func(varx=1, vary=2, *args):
        """"dummy function for testing"""
        output_list = [varx, vary]
        for arg in args:
            output_list += [arg]
        return output_list

    res0 = hlppy.help.call_with_kwargs(dummy_func,
                                         kwargs={})
    res1 = hlppy.help.call_with_kwargs(dummy_func,
                                         kwargs={'vary': 3})
    res2 = hlppy.help.call_with_kwargs(dummy_func,
                                         kwargs={}, args=[True, 'foo', 2])

    assert res0 == [1, 2]
    assert res1 == [1, 3]
    assert res2 == [True, 'foo', 2]

    del dummy_func
