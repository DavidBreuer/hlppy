"""Create fixtures for unit tests"""

# %%########################################################################### import modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# %%########################################################################### set parameters

# set seed
SEED = 0

# number of samples
LENY = 50

# number of features
LENX = 10

# number of target classes
LENC = 4

# %%########################################################################### define fixtures


# tmpdir_factory is used to create temporary directory, see:
# http://doc.pytest.org/en/latest/tmpdir.html#the-tmp-path-factory-fixture
# scope='session' so that a new directory is created only once per test session
@pytest.fixture(scope='session')
def temp_fake(tmpdir_factory):
    """Return temporary directory"""
    temp_fake = str(tmpdir_factory.mktemp('temp'))
    return temp_fake


@pytest.fixture
def x_fake():
    """Return fake x data"""
    np.random.seed(seed=SEED)
    x_fake = pd.DataFrame(np.random.rand(LENY, LENX))
    return x_fake


@pytest.fixture
def y_fake():
    """Return fake y data"""
    np.random.seed(seed=SEED)
    y_fake = pd.DataFrame([], index=range(LENY))
    y_fake['value'] = np.random.rand(LENY)
    y_fake['class'] = np.random.randint(0, LENC, size=LENY)
    return y_fake


@pytest.fixture
def imag_fake():
    """Return fake image data"""
    np.random.seed(seed=SEED)
    imag_fake = np.random.rand(30, 20)
    imag_fake[imag_fake > 0.95] = np.nan
    return imag_fake


# scope='function' so that a new axis is created for each test and
# modifications do not affect subsequent tests
@pytest.fixture(scope='function')
def plot_fake():
    """Return axis object"""
    plot_fake = plt.subplots(nrows=1, ncols=1)
    return plot_fake


# %%########################################################################### end file
