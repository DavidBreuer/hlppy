"""Test"""

import hlppy
import hlppy.help


def test_request():

    # set dummy token
    token = ''

    # check basic api page
    res = hlppy.help.request('', token)
    assert res is None

    # get specific label or missing token
    res = hlppy.help.request('labels', token)
    assert res is None
