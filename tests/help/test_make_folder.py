"""Test"""

import os
import tempfile

import hlppy
import hlppy.help


def test_make_folder():

    temp = tempfile.mkdtemp()
    temp = temp + '_postfix'
    assert not os.path.isdir(temp)

    out = hlppy.help.make_folder(temp)
    assert os.path.isdir(temp)
    assert out

    out = hlppy.help.make_folder(temp)
    assert os.path.isdir(temp)
    assert not out
