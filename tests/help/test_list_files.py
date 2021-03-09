"""Test"""

import os

import hlppy
import hlppy.help


def test_list_files():

    root = hlppy.help.get_project_root()

    # mostly default
    files = hlppy.help.list_files(root, directories=True)

    # sort
    assert sorted(files) == files
    # complete
    file = files[0]
    assert os.path.basename(file) != file
    # directories
    assert any([file.endswith('tests') for file in files])

    # opposite
    files = hlppy.help.list_files(root, sort=False, complete=False,
                                     inc='i', exc='x',
                                     directories=False)

    # no sort
    # assert sorted(files) != files
    # no complete
    file = files[0]
    assert os.path.basename(file) == file
    # no directories
    assert not any([file.endswith('tests') for file in files])
