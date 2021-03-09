"""Test"""

import hlppy
import hlppy.help


def test_git_version():

    tag = hlppy.help.git_version(main=True)
    assert isinstance(tag, str)

    tag = hlppy.help.git_version(main=False)
    assert isinstance(tag, str)
