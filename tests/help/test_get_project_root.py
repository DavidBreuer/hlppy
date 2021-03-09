"""Test"""

import pathlib

import hlppy
import hlppy.help


def test_get_project_root(y_fake):

    root = hlppy.help.get_project_root()

    assert isinstance(root, pathlib.PosixPath)

    # fails in CI where == 'app'
    # assert root.name == 'hlppy'
