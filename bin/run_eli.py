#!/usr/bin/env python
"""Run EliTech analysis."""

import sys

import hlppy
import hlppy.eli


if __name__ == '__main__':

    # get script arguments
    ARGS = sys.argv

    # set path
    path = '/media/Volume/Programs/elitech/'

    # download data and create plots
    hlppy.eli.download_data(path)
    tab = hlppy.eli.concatenate_data(path)
    hlppy.eli.plot_data(tab)

    # add linebreak
    print('')
