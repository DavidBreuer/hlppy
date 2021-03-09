"""Test"""

import os

import hlppy
import hlppy.help


def test_copy_function(temp_fake):

    def dummy(val):
        """Dummy function"""
        return val ** 2

    # check after copy
    copy = hlppy.help.copy_function(dummy)

    inp, res = 2, 4

    assert dummy(inp) == res
    assert copy(inp) == res   # pylint: disable=not-callable

    # check after saving to and loading from dump
    dump = os.path.join(temp_fake, 'dummy_func.npy')
    hlppy.help.save(dump, copy)
    copy = hlppy.help.load(dump)

    assert dummy(inp) == res
    assert copy(inp) == res

    # check after deletion of original function
    del dummy
    try:
        out = dummy(inp)  # noqa
    except Exception as err:
        out = err

    assert out != res
    assert isinstance(out, NameError)
    assert copy(inp) == res
