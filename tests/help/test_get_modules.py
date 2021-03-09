"""Test"""

import os

import pandas as pd

import hlppy
import hlppy.help


def test_get_modules():

    # from function
    tab = hlppy.help.get_modules('hlppy')
    mods = sorted(list(tab['module'].unique()))

    # from files
    root = hlppy.help.get_project_root()
    src = os.path.join(root, 'hlppy')
    files = hlppy.help.list_files(src, complete=False)
    files = [file.replace('.py', '') for file in files]

    # type
    assert isinstance(tab, pd.DataFrame)
    # size
    assert len(tab.columns) == 2
    # entries
    assert set(mods).issubset(files)
