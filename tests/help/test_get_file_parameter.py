"""Test"""

import hlppy
import hlppy.help


def test_get_file_parameter():
    """Test that extraction of parameters from file names works"""

    file = 'version=000-name=something.npy'

    assert hlppy.help.get_file_parameter(file, 'version') == '000'
    assert hlppy.help.get_file_parameter(file, 'name') == 'something'

    file = ''

    assert hlppy.help.get_file_parameter(file, 'version') is None
    assert hlppy.help.get_file_parameter(file, 'name') is None
