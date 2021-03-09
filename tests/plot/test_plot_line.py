import pytest

import hlppy
import hlppy.plot


@pytest.mark.mpl_image_compare(savefig_kwargs={'dpi': 90}, tolerance=1e-1)
def test_plot_line(mocker, plot_fake):

    # set variable
    steps = 3

    # pytest-mock must be installed for 'mocker' to be used!
    fig_fake, axis_fake = plot_fake
    mock_subplots = mocker.patch('hlppy.plot.plt.subplots')
    mock_subplots.return_value = (fig_fake, axis_fake)

    # without axis
    res = hlppy.plot.plot_line(steps, axis=None)
    assert res

    # with axis
    res = hlppy.plot.plot_line(steps, axis=axis_fake)
    assert res

    # important for decorator mpl_image_compare
    # test function must return a matplotlib figure object
    return fig_fake
