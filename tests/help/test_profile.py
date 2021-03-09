"""Test"""

import hlppy
import hlppy.help


def test_profile():
    """Test that profile returns expected output"""

    def dummy_func():
        """dummy function for testing"""
        return range(2000)

    profile_out = hlppy.help.profile(dummy_func)
    # test that profile returns a dictionary
    assert isinstance(profile_out, dict)
    # test that profile output has two keywords
    assert len(profile_out) == 2
    # keywords are 'tottime' and 'ncalls'
    keys = profile_out.keys()
    assert 'tottime' in keys
    assert 'ncalls' in keys
    # keywords are float / int
    assert isinstance(profile_out['tottime'], float)
    assert isinstance(profile_out['ncalls'], int)

    del dummy_func
