"""Define plot functions."""

# %%########################################################################### import modules

import matplotlib.pyplot as plt

# %%########################################################################### define functions


def plot_line(steps, axis=None):
    """Plot line (for unit testing).

    Parameters
    ----------
    steps : integer
        number of points in line
    axis : None or axis
        | None = create new axis
        | otherwise = existing matplotlib axis object

    Returns
    -------
    None : None
        create matplotlib figure and return None
    """
    if axis is None:
        _, axis = plt.subplots()
    axis.plot(range(steps))
    return True


# %%########################################################################### end module
