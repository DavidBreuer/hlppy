"""Test"""

import numpy as np

import hlppy
import hlppy.help


def test_nanmad(imag_fake):

    shape = imag_fake.shape

    out = hlppy.help.nanmad(imag_fake)

    assert isinstance(out, np.ndarray)
    assert out.shape == shape[:-1]
    assert np.isfinite(out).all()

    out = hlppy.help.nanmad(imag_fake, axis=0)
    assert out.shape == shape[1:]
