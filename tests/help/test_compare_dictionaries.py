"""Test"""

import copy

import hlppy
import hlppy.help


def test_compare_dictionaries():
    """Test that compare_dictionaries returns expected output"""

    dcta = dict(vala='a',
                valb='b',
                valc=dict(vala='A',
                          valb='B'))

    eql = hlppy.help.compare_dictionaries(dcta, dcta)
    assert eql

    dctb = copy.deepcopy(dcta)
    dctb['valb'] = 'x'

    eql = hlppy.help.compare_dictionaries(dcta, dctb)
    assert not eql

    dctb = copy.deepcopy(dcta)
    dctb['valc']['vala'] = 'X'

    eql = hlppy.help.compare_dictionaries(dcta, dctb)
    assert not eql
