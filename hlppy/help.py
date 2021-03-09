"""Define help functions"""

# %%########################################################################### import modules

import collections
import configparser
import cProfile
import datetime
import dill
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import math
import os
import pkg_resources
import pathlib
import psycopg2
import re
import requests
import subprocess
import sys
import textwrap
import time
import types
import warnings

import matplotlib.image
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import pulp
import scipy as sp
import sklearn
import sklearn.exceptions
import sklearn.metrics
import sklearn.model_selection
import stopit

# %%########################################################################### define functions


def example(arg):
    """Example function with PEP 484 type annotations.

    Details on doc string tests: https://stackoverflow.com/a/55002297/5350621

    Parameters
    ----------
    arg : float
        first parameter

    Returns
    -------
    res : float
        result, i.e., squared input

    Examples
    --------
    >>>
    'Do not doctest this part, but the following:'
    >>> example(9)
    81
    """
    res = arg**2
    return res


def angle(x, y=[1, 0]):
    """Compute angle [0, 360] between two vectors around origin.

    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors/879474

    Parameters
    ----------
    x : array
        2d array
    y : array
        2d array

    Returns
    -------
    ang : float
        angle

    Examples
    --------
    >>>
    m = 2*np.random.rand(1000, 2)-1
    c = [angle(x)/360 for x in m]
    plt.scatter(*m.T, c=c)
    """
    dot = -x[0]*y[0] + x[1]*y[1]
    det = -x[0]*y[1] - x[1]*y[0]
    ang = 270 - np.rad2deg(math.atan2(det, dot))
    return ang % 360


def circarrow(ax, diameter,centX, centY, startangle, angle,**kwargs):
    # https://stackoverflow.com/questions/44526103/matplotlib-draw-curved-arrow-that-looks-just-like-pyplot-arrow
    startarrow=kwargs.pop("startarrow",False)
    endarrow=kwargs.pop("endarrow",False)

    arc = matplotlib.patches.Arc([centX,centY],diameter,diameter,angle=startangle,
          theta1=np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if startarrow else 0,theta2=angle-(np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if endarrow else 0),linestyle="-",color=kwargs.get("color","black"))
    ax.add_patch(arc)

    if startarrow:
        startX=diameter/2*np.cos(np.radians(startangle)) + centX
        startY=diameter/2*np.sin(np.radians(startangle)) + centY
        startDX=+.000001*diameter/2*np.sin(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        startDY=-.000001*diameter/2*np.cos(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        ax.arrow(startX-startDX,startY-startDY,startDX,startDY,**kwargs)

    if endarrow:
        endX=diameter/2*np.cos(np.radians(startangle+angle)) + centX
        endY=diameter/2*np.sin(np.radians(startangle+angle)) + centY
        endDX=-.000001*diameter/2*np.sin(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        endDY=+.000001*diameter/2*np.cos(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        ax.arrow(endX-endDX,endY-endDY,endDX,endDY,**kwargs)
    return None


def git_version(main=True):
    """Determine the git tag of codebase and return it as a string.
    Change to project root folder to get correct git version.

    Based on: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92

    Why not just os.system('git describe --tags')?

    Parameters
    ----------
    main : Boolean
        return only main tag or also detailed tag description

    Returns
    -------
    tag : string
        git tag
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for key in ['SYSTEMROOT', 'PATH']:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    # get cwd
    cwd = os.getcwd()

    try:
        os.chdir(get_project_root())
        out = _minimal_ext_cmd(['git', 'describe', '--tags'])
        tag = out.strip().decode('ascii')
    except OSError:
        tag = 'unknown'

    # reset cwd
    os.chdir(cwd)

    # get only main tag without commit hash
    if main:
        tag = tag.split('-')[0]

    return tag


def compare_sets(veca, vecb, thres=30):
    """Save list of objects using dill to work with lambda functions etc.

    Parameters
    ----------
    veca : array or list or set
        set one
    vecb : array or list or set
        set two
    thres : int
        maximum number of items below which the items are printed as well

    Returns
    -------
    None : None
        None
    """

    seta = set(veca)
    setb = set(vecb)
    union = seta.union(setb)
    intersec = seta.intersection(setb)
    symdiff = seta.symmetric_difference(setb)
    diffab = seta.difference(setb)
    diffba = setb.difference(seta)

    # print stats
    ljust = 10
    print('len(a)'.ljust(ljust), len(veca))
    print('len{a}'.ljust(ljust), len(np.unique(veca)))
    print('len(b)'.ljust(ljust), len(vecb))
    print('len{b}'.ljust(ljust), len(np.unique(vecb)))

    # loop over set combinations
    # print lengths and items
    combis = ['union', 'intersec', 'symdiff', 'diffab', 'diffba']
    for combi in combis:
        comb = np.sort(list(eval(combi)))
        lenc = len(comb)
        print(combi.ljust(ljust), lenc)
        if lenc < thres:
            print(comb)
        else:
            print('[too many]')

    return union, intersec, symdiff, diffab, diffba


def compare_dataframes(taba, tabb, dummy='dummy'):
    """Compare two tables, check that all non-NaN entries match,
    and print deviations.

    Parameters
    ----------
    taba : array or dataframe
        input data one
    taba : array or dataframe
        input data two
    dummy : string
        dummy string to replace NaN values

    Returns
    -------
    eql : Boolean
        input tables are identical or not
    """

    # copy inputs
    cpya = pd.DataFrame(taba).copy()
    cpyb = pd.DataFrame(tabb).copy()

    # check shapes
    eql = (cpya.shape == cpyb.shape)
    if not eql:
        print('Shape mismatch')
        return eql

    # check if all NaNs agree
    nan = (cpya.isna() == cpyb.isna())
    eqn = np.all(nan)

    # fill NaNs
    cpya = cpya.fillna(dummy)
    cpyb = cpyb.fillna(dummy)

    # check if all values agree
    val = (cpya == cpyb)
    eqv = np.all(val)

    # plot mismatches
    eql = eqn and eqv
    if not eql:
        print('Value mismatch')
        ims = pd.DataFrame(np.full(taba.shape, 0))
        ims[~nan] = -1
        ims[nan & ~val] = 1
        plt.title(ims.shape)
        plt.imshow(ims, vmin=-1, vmax=1,
                   aspect='auto', cmap='PiYG', origin='lower')
        ticks = [-1, 0, 1]
        cbar = plt.colorbar(ticks=ticks)
        yticklabels = ['NaN mismatch', 'identical', 'value mismatch']
        cbar.ax.set_yticklabels(yticklabels)

    return eql


def compare_dictionaries(dcta, dctb, level=None, quiet=False):
    """Compare dictionaries recursively.

    Parameters
    ----------
    dcta : dictionary
        input data one
    dctb : dictionary
        input data two
    level : None or string
        parent hierarchy levels
    quiet : Boolean
        print status messages by default

    Returns
    -------
    eql : Boolean
        input dictionaries are identical or not
    """
    eqls = []
    for key in dcta.keys():
        vala = dcta[key]
        valb = dctb[key]
        if isinstance(vala, dict):
            eql = compare_dictionaries(vala, valb, level=key, quiet=quiet)
        else:
            eql = np.all(vala == valb)
            if level:
                key = '%s -> %s' % (level, key)
            if not quiet:
                print(key, eql)
        eqls.append(eql)
    eql = np.all(eqls)
    return eql


@stopit.threading_timeoutable(default=None)
def stop_example(iterations):
    """Function demonstrating the usage of the stopit timeout decorator.

    Parameters
    ----------
    iterations : int
        number of seconds the functions runs

    Returns
    -------
    iteration : int
        current iterction

    Examples
    --------
    >>> stop_example(1, timeout=2)
    1
    >>>
    out = stop_example(3, timeout=2)
    None
    """
    for iteration in range(1, 1 + iterations):
        time.sleep(1)
    return iteration


def save(file, obj):
    """Save list of objects using dill.

    Parameters
    ----------
    file : string
        output file name
    obj : object
        objects, e.g. array, lists, functions etc.

    Returns
    -------
    None : None
        None
    """
    with open(file, 'wb') as dump:
        dill.dump(obj, dump)
    return None


def load(file):
    """Load list of objects using dill.

    Parameters
    ----------
    file : string
        input file name

    Returns
    -------
    obj : object
        object, e.g. array, lists, functions etc.
    """
    with open(file, 'rb') as dump:
        obj = dill.load(dump)
    return obj


def count_elements(vec):
    """Count elements in list and return keys and counts.

    Parameters
    ----------
    vec : list
        list

    Returns
    -------
    xval : list
        unique list entries
    yval : list
        frequency of list entries
    """
    freq = collections.Counter(vec)
    xval = np.unique(vec)
    yval = [freq[key] for key in xval]
    return xval, yval


def get_datetime(inp):
    """Format given datetime or return current date and time according to ISO.

    Parameters
    ----------
    inp : None or datetime
        | None = return current datetime
        | inp = format given datetime object

    Returns
    -------
    temp : string
        datetime string in ISO format %Y%m%d-%H%M%S
    """
    if inp is None:
        inp = datetime.datetime.now()
    temp = inp.strftime('%Y%m%d-%H%M%S')
    return temp


def is_fitted(model):
    """Check if sklearn model has already been fitted.

    Parameters
    ----------
    model : sklearn model
        arbitrary sklearn model

    Returns
    -------
    out : Boolean
        return true or false
    """

    # https://stackoverflow.com/a/39900933/5350621
#    out = True
#    try:
#        model.predict(None)
#    except sklearn.exceptions.NotFittedError:
#        out = False

    # https://stackoverflow.com/a/48046685/5350621
    attr = [key for key, val in inspect.getmembers(model)
            if key.endswith('_') and not key.startswith('__')]
    out = attr > 0

    return out


def scale(array, vmin=None, vmax=None, invert=False, norm=False,
          unitize=False):
    """Scale array to unit interval with or without QuantileTransformation
    or invert scaling.

    Parameters
    ----------
    array : array
        multi-dimensional array
    vmin : None or float
        minimum is scaled to 0
        None = take array minimum
    vmax : None or float
        maximum is scaled to 1
        None = take array maximum
    invert : Boolean
        use min/max values to invert scaling
    norm : Boolean
        normalize sum of input to one
    unitize : Boolean
        transform input data to unit interval before rescaling using
        sklearn QuantileTransformer, does not work with invert parameter!

    Returns
    -------
    out : array
        rescaled array
    """
    if invert:
        out = array * (vmax - vmin) + vmin
    else:
        if unitize:
            tra = sklearn.preprocessing.QuantileTransformer()
            array = tra.fit_transform(np.atleast_2d(array).T)[:, 0]
        if vmin is None:
            vmin = np.nanmin(array)
        if vmax is None:
            vmax = np.nanmax(array)
        # avoid division-by-zero error
        if vmax == vmin:
            vmin, vmax = 0, 1
        out = (array - vmin) / (vmax - vmin)
    if norm:
        out = out/np.nansum(out)
    return out


def copy_function(func, name=None):
    """Return a function with same code, globals, defaults, closure, and
    name (or provide a new name). Code taken from:

    https://stackoverflow.com/a/30714299/5350621

    Parameters
    ----------
    func : function
        original function to be copied
    name : None or string
        take old or set new function name

    Returns
    -------
    cpy : array
        copied function
    """
    cpy = types.FunctionType(func.__code__,
                             func.__globals__,
                             name or func.__name__,
                             func.__defaults__,
                             func.__closure__)
    # in case func was given attrs (note this dict is a shallow copy)
    cpy.__dict__.update(func.__dict__)
    return cpy


def stdout_to_string(func, *args, **kwargs):
    """Capture printed stdout when calling given function and return it as
    string.
    https://wrongsideofmemphis.com/2010/03/01/store-standard-output-on-a-variable-in-python/

    Parameters
    ----------
    func : function
        function with printed stdout
    args : list, optional
        positional argument(s) of function func
    kwargs : list, optional
        keyword argument(s) of function func

    Returns
    -------
    result : string
        string of printed stdout
    """

    # get original system stdout
    original_stdout = sys.stdout
    # create StringIO object
    # use try-catch to make compatible with Python 2 and 3
    try:
        # StringIO is not available in python3
        # disable=import-error ignores that for pylint>=0.10
        # see: https://stackoverflow.com/a/39348054/9305415
        import StringIO  # pylint: disable=import-error
        captured_output = StringIO.StringIO()
    except ImportError:
        captured_output = io.StringIO()
    # redirect stdout
    sys.stdout = captured_output
    # call function func
    func(*args, **kwargs)
    # IMPORTANT! reset original system stdout
    sys.stdout = original_stdout
    # return string
    result = captured_output.getvalue()
    return result


def get_project_root() -> pathlib.Path:
    """Return project root folder.

    Function definition uses Type Hints:
    https://www.python.org/dev/peps/pep-0484/

    Parameters
    ----------
    None : None
        None

    Returns
    -------
    proj : string
        project root folder, e.g., '/home/david/Repos/predict'
    """
    # https://stackoverflow.com/a/53465812/5350621
    proj = pathlib.Path(__file__).parent.parent
    return proj


def insert_string(string, value, location):
    """Insert substring into string at given location.

    Parameters
    ----------
    out : string
        input string
    value : string
        input substring
    location : int
        index of insertion location

    Returns
    -------
    out : string
        output string
    """
    out = string[:location] + value + string[:location]
    return out


def get_envs(section):
    """Read section in environment variable file .env from project root folder
    containing database access data of the form:

    [section]
    name =
    host =
    user =
    word =

    Parameters
    ----------
    section : string
        section in .env file, e.g., [section]

    Returns
    -------
    name : string
        database name
    host : string
        database host IP
    user : string
        database user name
    word : string
        database password
    """
    # https://docs.python.org/3/library/configparser.html
    proj = get_project_root()

    config = configparser.ConfigParser()
    config.read(os.path.join(proj, '.env'))

    name = config.get(section, 'name', fallback=None)
    host = config.get(section, 'host', fallback=None)
    user = config.get(section, 'user', fallback=None)
    word = config.get(section, 'word', fallback=None)

    return name, host, user, word


def get_location(path, file, package='hlppy'):
    """Get location of package data.

    Parameters
    ----------
    path : string
        data path in package, e.g., 'data'
    file : string
        data file name, e.g., 'logo.png'
    package : string
        package name

    Returns
    -------
    name : string
        full file path
    """
    loca = os.path.join(path, file)
    name = pkg_resources.resource_filename(package, loca)
    return name


def list_location(path, package='snippets'):
    """List files in data location of package.

    Parameters
    ----------
    path : string
        data path in package
    package : string
        package name

    Returns
    -------
    names : string
        full file paths
    """
    loca = pkg_resources.resource_filename(package, path)
    files = sorted(os.listdir(loca))
    names = [os.path.join(loca, file) for file in files]
    return names


def filter_columns(tab, inc=None, exc=None, table=True):
    """Filter data frame columns by strings which should be included or
    excluded.

    Parameters
    ----------
    tab : data frame
        input data frame
    inc : None or string
        | include columns whose name contains this string
        | may contain multiple strings separated by |
    exc : None or string
        | exclude columns whose name contains this string
        | may contain multiple strings separated by |
    table : Boolean
        return filtered table or column names only

    Returns
    -------
    out : data frame
        output data frame
    """
    cpy = tab.copy()
    if inc:
        cols = [col for col in cpy.columns
                if any([val in col for val in inc.split('|')])]
        cpy = cpy[cols]
    if exc:
        cols = [col for col in cpy.columns
                if all([val not in col for val in exc.split('|')])]
        cpy = cpy[cols]
    if table:
        out = cpy[cols]
    else:
        out = cols
    return out


def detect_elbow(vec):
    """Detect elbow in 1D curve as point of minimum curvature. Can be used to
    determine optimal number of components in PCA.

    Parameters
    ----------
    vec : array
        one-dimensional array

    Returns
    -------
    idx : integer
        position of elbow
    """
    # detect elbow via minimum curvature,
    # for uniformly spaced vectors only
    curv = np.gradient(np.gradient(vec))
    idx = np.nanargmin(curv)
    return idx


def convert_labels_classes(lab):
    """Convert array of binary labels to integer classes.

    Parameters
    ----------
    lab : array
        binary labels

    Returns
    -------
    cla : array
        integer classes
    """
    cla = np.array([convert_labels_string(val)
                    for key, val in lab.iterrows()])
    return sklearn.preprocessing.LabelEncoder().fit_transform(cla)


def convert_labels_string(array):
    """Convert array of binary labels to string.

    Parameters
    ----------
    array : array
        binary labels

    Returns
    -------
    string : string
        binary string
    """
    string = ''.join(array.astype(np.str))
    return string


def call_with_kwargs(func, kwargs, args=None):
    """Call function with parameters and keyword arguments.

    Parameters
    ----------
    func : function
        function to call
    kwargs : args
        keyword arguments
    args : args
        regular arguments

    Returns
    -------
    func_new : function
        function with parameters and keyword arguments
    """
    allowed = inspect.getfullargspec(func).args  # pylint: disable=no-member
    dicts = {key: val for key, val in kwargs.items() if key in allowed}
    if args is None:
        func_new = func(**dicts)
    else:
        func_new = func(*args, **dicts)
    return func_new


def profile(func, sort='tottime'):
    """Print profile of function.

    Excute function for a number of iterations and print profiling report

    Parameters
    ----------
    func : function
        function without parameter
    sort : {'tottime', 'cumtime'}
        profiling report

    Returns
    -------
    None

    Examples
    --------
    >>>
    def func(): sum(range(10000))
    profile(func)['ncalls']
    3
    """
    prof = cProfile.Profile()
    prof.enable()
    func()
    prof.disable()
    report = stdout_to_string(prof.print_stats, sort=sort)

    out = {}
    ifunc = report.find("function")
    iin = report.find("in") + 2
    isec = report.find("seconds")
    out["ncalls"] = int(report[0:ifunc])
    out["tottime"] = float(report[iin:isec])
    return out


def make_folder(path):
    """Create folder if non-existent.

    Parameters
    ----------
    path : string
        directory

    Returns
    -------
    res : Boolean
        return success or failure
    """
    res = True
    try:
        os.makedirs(path)
    except OSError:
        res = False
    return res


def insert_key_value(dct, pos, key, val):
    """Insert key-value pair into dictionary at given position.

    Parameters
    ----------
    dct : dictionary
        dictionary
    pos : integer
        index in dictionary, positive or negative integer or string
    key : string
        key name of new entry
    val : object
        value of new entry

    Returns
    -------
    new : dictionary
        new, extended dictionary
    """
    # if key exists, overwrite data
    if key in dct.keys():
        new = {ikey: ival for ikey, ival in dct.items()}
        new[key] = val
        return new

    if isinstance(pos, str):
        # if position is string, insert after that key
        pos = np.argmax([pos == kei for kei in dct.keys()])
        poz = pos + 1
    elif pos < 0:
        # if position is negative, insert at position from end
        poz = len(dct) + pos
    else:
        poz = pos

    # create new dictionary
    new = dict()
    for idx, (ikey, ival) in enumerate(dct.items()):

        # if position is positive, insert before
        if pos >= 0 and idx == poz:
            new[key] = val
        new[ikey] = ival

        # if position is negative, insert after
        if pos < 0 and idx == poz:
            new[key] = val

    return new


def insert_key_value_inplace(params1, pos, params2, key2):
    """Convenience function to insert value of second dictionary into first.

    Parameters
    ----------
    params1 : dictionary
        target dictionary
    pos : integer
        index in dictionary, must be smaller than length of dictionary
    params2 : dictionary
        dictionary from which data are copied
    key2 : string
        key name in second dictionary from which data are copied

    Returns
    -------
    new : dictionary
        new, extended dictionary
    """
    new = insert_key_value(params1, pos,
                                        key2, params2[key2])
    return new


def nanmad(array, axis=-1):
    """Compute median absolute deviation (MAD) of array along given axis.

    Parameters
    ----------
    array : array
        array
    axis : integer
        axis along which MAD is computed

    Returns
    -------
    out : array
        MAD of array along given axis
    """
    out = np.nanmedian(np.abs(array - np.nanmedian(array,
                                                   axis=axis,
                                                   keepdims=True)), axis=axis)
    return out


def nanptp(array, axis=-1):
    """Compute peak-to-peak (PTP) distance between minimum and maximum of array
    along given axis.

    Parameters
    ----------
    array : array
        array
    axis : integer
        axis along which MAD is computed

    Returns
    -------
    out : array
        PTP of array along given axis
    """
    out = np.nanmax(array, axis=axis) - np.nanmin(array, axis=axis)
    return out


def naniqr(array, axis=-1):
    """Compute interquartile range (IQR) of array along given axis.

    Parameters
    ----------
    array : array
        array
    axis : integer
        axis along which MAD is computed

    Returns
    -------
    out : array
        IQR of array along given axis
    """
    out = np.subtract(*np.nanpercentile(array, [75, 25], axis=axis))
    return out


def nancv(array, axis=-1, robust=True):
    """Compute coefficient of variation (CV) of array along given axis.

    Parameters
    ----------
    array : array
        array
    axis : integer
        axis along which MAD is computed
    robust : Boolean
        use robust median/iqr of data or regular avg/std

    Returns
    -------
    out : array
        CV of array along given axis
    """
    if robust:
        out = naniqr(array, axis=axis) / np.nanmedian(array, axis=axis)
    else:
        out = np.nanstd(array, axis=axis) / np.nanmean(array, axis=axis)
    return out


def nancorr(array, vector):
    """Compute (Pearson) correlation coefficient between vector and array
    for each row. This is much faster than available numpy/scipy alternative
    when vector and array are given.

    Parameters
    ----------
    array : array
        input array
    vector : array
        vector to correlate array rows with

    Returns
    -------
    out : array
        correlation coefficients of array rows with vector
    """
    prod = (array.T - np.nanmean(array, axis=1)).T * (vector - np.nanmean(vector))
    upper = np.nanmean(prod.T, axis=0)
    lower = np.nanstd(array, axis=1) * np.nanstd(vector)
    out = upper / lower
    return out


def nanargmin(array, axis=-1):
    """Compute nanargmin along given axis. If one array shape is zero,
    set to one to avoid ValueError 'argmin of an empty sequence'.

    Parameters
    ----------
    array : array
        input array
    axis : integer
        axis along which nanargmin is computed

    Returns
    -------
    idx : array
        index of minimum along given axis
    """
    shape = np.array(array.shape)
    # if array is empty along some axes, create dummy array to avoid ValueError
    if min(shape) == 0:
        shape[shape == 0] = 1
        new = np.zeros(shape)
    else:
        new = array

    # to avoid all-nan ValueError (for 2-dimensional with axis=1)
    idnan = np.all(np.isnan(new), axis=axis)
    idx = np.argmin(new, axis=axis)
    idx[~idnan] = np.nanargmin(new[~idnan], axis=axis)

    print(idx)

    return idx


def vrange(vec):
    """Generate sequence of length of given vector, similar to convenient 'seq'
    function in R.

    Parameters
    ----------
    vec : iterable object
        list or array or other

    Returns
    -------
    out : list
        sequence of same length as input
    """
    return range(len(vec))


def irange(vmin, vmax, vinc):
    """Auxiliary function for creating integer range, also for float input.

    Parameters
    ----------
    vmin : int or float
        lower boundary
    vmax : int or float
        upper boundary
    vinc : int or float
        step size

    Returns
    -------
    lst : list
        list of integer range
    """
    rng = np.arange(vmin, vmax, vinc).astype(np.int)
    lst = list(rng)
    return lst


def get_file_parameter(file, para):
    """Get parameter from file name.

    Parameters
    ----------
    file : string
        file name
    para : string
        name of parameter of interest

    Returns
    -------
    val : string
        parameter value
    """
    base = '.'.join(os.path.basename(file).split('.')[:-1])
    lst = [bas.split('=')[-1] for bas in base.split('-') if para + '=' in bas]
    if not lst:
        return None
    return lst[0]


def approx(values, targets):
    """Get indices of target values to which original values are closest,
    similar to convenient 'approx' function in R.

    Parameters
    ----------
    values : array
        original values
    targets : array
        target values

    Returns
    -------
    indices : array
        indices of target values to which original values are closest
    """
    # https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    tree = sp.spatial.cKDTree(values)  # pylint: disable=no-member
    _, indices = tree.query(targets)
    return indices


def round_to(array, step=1):
    """Round array to given integer steps, used to bin sensor wavelengths.

    Parameters
    ----------
    array : array
        original array
    step : integer
        integer rounding steps

    Returns
    -------
    out : array
        rounded array
    """
    return np.round(array / step) * step


def join_columns(tab, cols, sep='.'):
    """Join multiple columns of data frame with separator and convert to string
    if necessary.

    Parameters
    ----------
    tab : data frame
        original data
    cols : list of strings
        names of original columns to be joined
    sep : string
        symbol used to join columns

    Returns
    -------
    cpy : data series
        new data frame column
    """
    cpy = pd.Series('', index=tab.index)
    for idc, col in enumerate(cols):
        seq = sep * (idc > 0)  # only add separator after first step
        cpy = cpy + seq + tab.loc[:, col].map(str)
    return cpy


def get_minmax(lst, extend=0):
    """Get minimum and maximum of list of arrays, and extend by fraction
    of range.

    Parameters
    ----------
    lst : list of array
        original data
    extend : float
        fraction of range that is added to min/max, e.g.,
        convenient for plotting

    Returns
    -------
    vmin : float
        (extended) minimum
    vmax : float
        (extended) minimum
    """
    lst = np.hstack([np.hstack(sub) for sub in lst])
    vmin, vmax = np.nanmin(lst), np.nanmax(lst)
    vext = extend * (vmax - vmin)
    return vmin - vext, vmax + vext


def gridify_parameters(params, parameters=None):
    """Create dictionary of parameter combinations for full grid search.

    Parameters
    ----------
    params: dictionary
        dictionary with parameters
    parameters : list
        list of parameter dictionaries/values

    Returns
    -------
    parameters : list
        list of parameter dictionaries/values
    """

    # prepare list to add paramters to when going recursively through input
    if parameters is None:
        parameters = []

    if isinstance(params[next(iter(params))], dict):
        # if first element of input dict is still a dict,
        # apply function recursively
        for param in params.values():
            parameters = gridify_parameters(param, parameters=parameters)
    else:
        # otherwise create grid of parameter values and append to list
        keys = list(params.keys())
        vals = list(params.values())
        for prods in itertools.product(*vals):
            tpl = [(keys[idp], prod) for idp, prod in enumerate(prods)]
            dct = dict(tpl)
            parameters.append(dct)

    return parameters


def make_name(dct, exclude=None, include=None, limit=None):
    """Return input dictionary as string for filename.

    Parameters
    ----------
    dct : dictionary
        dictionary of keyword arguments
    exclude : list, optional
        list of arguments that are ignored in args
    include : list, optional
        list of arguments that are included
    limit : None or integer
        limit length of keys in output string,
        not recommended as it may cause non-unique file names

    Returns
    -------
    string : string
        output string

    Examples
    --------
    >>> make_name(dict(a=1, b='blub'), exclude=['a'])
    'b=blub'
    """
    cpy = dct.copy()
    if include is None:
        if exclude is None:
            exclude = []
        for exc in exclude:
            cpy.pop(exc, None)
    else:
        new = dict()
        for inc in include:
            new[inc] = cpy.get(inc)
        cpy = new
    # remove dashes in value
    # if value contains slashes use only last part
    # limit length of key string to avoid 'OSError: File name too long'
    # https://stackoverflow.com/q/34503540/5350621
    string = '-'.join([k[:limit] + '=' + str(v).split('/')[-1].replace('-', '')
                       for k, v in cpy.items()])
    return string


def is_unique(lst):
    """Check if list contains only unique values, no duplicates.

    Parameters
    ----------
    out : list
        target list

    Returns
    -------
    uni : Boolean
        list contains only unique values or not
    """
    out = len(lst) == len(set(lst))
    return out


def is_identical(lst):
    """Check if list contains only single/identical value, all duplicates.

    Parameters
    ----------
    out : list
        target list

    Returns
    -------
    uni : Boolean
        list contains only identical values or not
    """
    out = len(set(lst)) == 1
    return out


def is_included(strngs, lsts, checkmode=any):
    """Check if any string is included in any of list items.

    Parameters
    ----------
    strngs : string or list
        string or strings of interest
    lsts : list
        target list
    checkmode : function
        choose if 'any' or 'all' strings must be contained in list items

    Returns
    -------
    out : list
        Boolean list
    """
    if isinstance(strngs, str):
        strngs = [strngs]
    out = [checkmode([strng in lst for strng in strngs]) for lst in lsts]
    return out


def flatten_list(lst):
    """Flatten nested list, modified from:
    https://stackoverflow.com/a/952952

    Parameters
    ----------
    lst : list
        target list

    Returns
    -------
    out : list
        unnested list
    """
    out = [item for sublist in lst for item in sublist]
    return out


def bin_list(lst, steps=8):
    """Bin list of continuous values with nans into given number of categories.

    Parameters
    ----------
    lst : list
        target list
    steps : integer
        number of output bins

    Returns
    -------
    categ : list
        list of categories
    """
    bins = np.nanpercentile(lst, np.linspace(0, 100, steps))
    digi = np.digitize(lst, bins)
    digi[np.isnan(lst)] = 0
    categ = digi.astype(np.str)
    return categ


def clean_none(ele):
    """Clean elements to make them sortable, i.e. convert None and tuple to
    string.

    Parameters
    ----------
    ele : element
        element

    Returns
    -------
    add : element
        cleaned element
    """
    if ele is None:
        add = 'None'
    elif isinstance(ele, tuple):
        add = str(ele)
    else:
        add = ele
    return add


def sort_natural(lst):
    """Sort strings with number in natural order. Modified from:
    https://stackoverflow.com/a/4836734

    Parameters
    ----------
    lst : list
        input list of strings

    Returns
    -------
    srt : list
        sorted list

    Examples
    --------
    lst = ['imp_10', 'imp_100', 'imp_101', 'imp_12']
    sort_natural(lst)
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    ankey = lambda key: [convert(res) for res in re.split('([0-9]+)', key)]
    srt = sorted(lst, key=ankey)
    return srt


def sort_none(lst):
    """Auxiliary function to allow sorting of None and strings:
    https://stackoverflow.com/a/12971697/5350621

    Parameters
    ----------
    lst : list
        input list

    Returns
    -------
    out : list
        sorted list
    order : list
        sort order

    Examples
    --------
    lst = [12, 4, 8, 'lv', None]
    sort_none(lst)
    """
    # clean input from nans and convert tuples to strings etc.
    new = [clean_none(ele) for ele in lst]
    # sort numbers and strings
    out = natsort.natsorted(new)
    # reobtain sorting order
    order = natsort.index_natsorted(new)
    return out, order


def sort_dictionary(dct, which='key'):
    """Sort dictionary by key or value.

    Parameters
    ----------
    dct : dictionary
        input dictionary
    which : string
        sort by 'key' or 'value'

    Returns
    -------
    cpy : dictionary
        sorted dictionary
    """
    cpy = dict()
    items = list(dct.items())
    whi = 1*(which == 'value')
    items.sort(key=lambda ele: ele[whi])
    for key, val in items:
        cpy[key] = dct[key]
    return cpy


def invert_dictionary(dct):
    """Switch keys and values of dictionary.

    Parameters
    ----------
    dct : dictionary
        input dictionary

    Returns
    -------
    cpy : dictionary
        inverted dictionary
    """
    cpy = dict()
    for key, val in dct.items():
        cpy[val] = key
    return cpy


def append_dictionary(dinp, dapp, listed=True):
    """Append entries of second dictionary to those of first, append
    to list if the key is already in use.

    Parameters
    ----------
    dinp : dictionary
        first dictionary
    dapp : dictionary
        second dictionary
    listed : Boolean
        True = append entries to list, if key is already in use
        False = overwrite entry, if key is already in use

    Returns
    -------
    dout : dictionary
        new, extended dictionary
    """
    dout = dinp.copy()
    if listed:
        for key, _ in dapp.items():
            old = dout.get(key)
            if not isinstance(old, list):
                if old is None:
                    dout[key] = []
                else:
                    dout[key] = [dout.get(key)]
            dout[key].append(dapp[key])
    else:
        for key, _ in dapp.items():
            dout[key] = dapp[key]
    return dout


def get_partial_string(string, start=None, stop=None, sep='-'):
    """Get only part of string which is divided by separator.

    Parameters
    ----------
    string : string
        string
    start : None or integer
        | None = take string parts from beginning
        | integer = take string parts from this position
    stop : None or integer
        | None = take string parts until
        | integer = take string parts up to this position
    sep : string
        separator which divides the string into parts
    Returns
    -------
    res : string
        partial string
    """
    vec = string.split(sep)[start:stop]
    res = sep.join(vec)
    return res


def wrap_string(string, mode='dict', width=60):
    """Wrap string to fit given length, useful for plotting titles.

    Parameters
    ----------
    string : object
        string or dictionary or arbitrary object that is converted to string
    mode : None or string
        None = wrap string after given width
        'dict' = break string after each dictionary key
    width : integer
        wrap after given number of characters

    Returns
    -------
    out : string
        wrapped string
    """
    # string = 'asdsd asd sdgs dfdsf sdf sfasd asd fsdf sdf asd dsf sdf sdf'
    # https://stackoverflow.com/a/48860937
    string = str(string)
    if mode == 'dict':
        out = string.replace(',', '\n')
    else:
        out = '\n'.join(textwrap.wrap(string, width))
    return out


def list_files(folder, sort=True, complete=True, inc=None, exc=None,
               directories=None):
    """List files in folder.

    Parameters
    ----------
    folder : string
        folder of interest
    sort : Boolean
        sort files
    complete : Boolean
        return complete file names
    inc : None or string
        include files whose name contains this string
    exc : None or string
        exclude files whose name contains this string
    directories : None or Boolean
        | None = return files and directories
        | False = return files only
        | True = return directories only

    Returns
    -------
    files : list
        list of file names
    """

    files = os.listdir(folder)
    if directories is False:
        files = [file for file in files
                 if os.path.isfile(os.path.join(folder, file))]
    elif directories is True:
        files = [file for file in files
                 if os.path.isdir(os.path.join(folder, file))]
    if inc:
        files = [file for file in files if inc in file]
    if exc:
        files = [file for file in files if exc not in file]
    if sort:
        files = sorted(files)
    if complete:
        files = [os.path.join(folder, file) for file in files]
    return files


def get_file_datetime(file):
    """Get file modification datetime.

    Returns
    -------
    file : string
        file name
    """
    tsp = os.path.getmtime(file)
    dtm = datetime.datetime.fromtimestamp(tsp)
    res = get_datetime(inp=dtm)
    return res


def compute_distance2line(ppp, pst, psp):
    """Compute distances between point and line.

    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

    Parameters
    ----------
    ppp : array
        set of two-dimensional points
    pst : array
        two-dimensional start point
    psp : array
        two-dimensional stop point

    Returns
    -------
    dist : array
        distances to line
    line : array
        array of line for plotting
    """
    dist = np.cross(psp-pst, ppp-pst)/np.linalg.norm(psp-pst)
    line = np.array([pst, psp]).T
    return dist, line


# %%########################################################################### modules


def get_modules(package):
    """Get all functions of all modules of a package and return as data frame.

    https://stackoverflow.com/questions/139180/how-to-list-all-functions-in-a-python-module
    https://stackoverflow.com/questions/8718885/import-module-from-string-variable
    https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
    https://stackoverflow.com/questions/218616/getting-method-parameter-names-in-python/

    Parameters
    ----------
    package : string
        module

    Returns
    -------
    tab : data frame
        table of module and function names
    """

    pkg = importlib.import_module(package)
    mods = [mod for mod in dir(pkg) if '_' not in mod and 'Not' not in mod]

    lst = []
    for mod in mods:
        pkg = importlib.import_module('%s.%s' % (package, mod))
        fncs = [fnc[1] for fnc in inspect.getmembers(pkg)
                if inspect.isfunction(fnc[1])]
        for fnc in fncs:
            nam = fnc.__name__ + str(inspect.signature(fnc))
            lst.append([mod, nam])

    tab = pd.DataFrame(lst, columns=['module', 'function'])

    return tab


# %%########################################################################### hashes


def get_hash(string, limit=None):
    """Compute SHA256 hash of string, clip if needed.

    Parameters
    ----------
    string : string
        input string
    limit : None or integer
        return only given number of hash digits/symbols

    Returns
    -------
    hsh : string
        hash of string in hex format
    """
    string = str(string)
    # https://stackoverflow.com/a/42089311
    hsh = hashlib.sha256(string.encode('utf-8')).hexdigest()[:limit]
    return hsh



# %%########################################################################### benchmark


def timeit(func, args=None, reps=10):
    """Execute function and measure time.

    Parameters
    ----------
    func : function
        function of interest
    reps : integer
        number of repetitions

    Returns
    -------
    result : object
        result of (final) function execution
    durations : list
        durations of repeated function executions
    """
    durations = [np.nan]*reps
    for rep in range(reps):
        temp = time.time()
        if args is None:
            result = func()
        else:
            result = func(*args)
        durations[rep] = time.time() - temp
    return result, durations


def benchmark(*funcs, args=None, reps=10, check=True, factor=None):
    """Perform benchmark of given functions and check equality of results.

    Parameters
    ----------
    funcs : list of functions
        functions of interest
    reps : integer
        number of repetitions
    check : Boolean
        check that function return identical results
    factor : None or array
        effective speed to incorporate differences if some of the functions
        are vectorized and include multiple repetitions already,
        if in doubt do not use this!

    Returns
    -------
    names : list
        function names, e.g., for plotting
    results : object
        result of (final) function executions
    durations : list
        durations of repeated function executions
    """
    lenf = len(funcs)
    results = [[]]*lenf
    durations = [[]]*lenf
    names = [[]]*lenf
    label = ['avg', 'std', 'med', 'iqr', 'ratio']
    ljust = 20
    for idf, func in enumerate(funcs):
        names[idf] = func.__name__
        results[idf], durations[idf] = timeit(func, args=args, reps=reps)
        if factor is not None:
            # rescale duration to account for vectorized inputs
            durations[idf] = np.divide(durations[idf], factor[idf])
        avg = np.nanmean(durations[idf])
        std = np.nanstd(durations[idf])
        med = np.nanmedian(durations[idf])
        iqr = naniqr(durations[idf])
        ratio = np.nanmean(durations[idf]) / np.nanmean(durations[0])
        vec = [avg, std, med, iqr, ratio]
        stats = ', '.join([label[idv] + '=%.1e' % val
                           for idv, val in enumerate(vec)])
        print(names[idf].ljust(ljust), ':', stats)
    # check results
    if check:
        eqs = [np.allclose(result, results[0]) for result in results]
        if not np.all(eqs):
            print('Functions do not return identical results. '
                  'Results deviating from first one:',
                  np.array(names)[eqs])
    return names, results, durations


# %%########################################################################### sample


def sample_fraction(vec, fraction):
    """Return random sample of input, without replacement.

    Parameters
    ----------
    vec : int or list
        original data, for integer inputs a list of this length is created,
        for convenience
    fraction : float
        fraction of random element to return

    Returns
    -------
    samp : list
        random sample of input
    """
    if isinstance(vec, np.int):
        frac = vec*fraction
    else:
        frac = len(vec)*fraction
    samp = np.random.choice(vec, int(frac), replace=False)
    return samp


def list_sub(lsta, lstb):
    """Subtract two lists of arrays.

    Parameters
    ----------
    lsta : list
        first list
    lstb : list
        second list

    Returns
    -------
    lst : lst
        result
    """
    # auxiliary functions for arithmetics of list of arrays
    return [np.subtract(lsta[idi], lstb[idi]) for idi, lst in enumerate(lsta)]


def list_add(lsta, lstb):
    """Add two lists of arrays.

    Parameters
    ----------
    lsta : list
        first list
    lstb : list
        second list

    Returns
    -------
    lst : lst
        result
    """
    return [np.add(lsta[idi], lstb[idi]) for idi, lst in enumerate(lsta)]


def list_mul(lsta, lstb):
    """Multiply two lists of arrays.

    Parameters
    ----------
    lsta : list
        first list
    lstb : list
        second list

    Returns
    -------
    lst : lst
        result
    """
    return [np.multiply(lsta[idi], lstb[idi]) for idi, lst in enumerate(lsta)]


def list_div(lsta, lstb):
    """Divide two lists of arrays.

    Parameters
    ----------
    lsta : list
        first list
    lstb : list
        second list

    Returns
    -------
    lst : lst
        result
    """
    return [np.divide(lsta[idi], lstb[idi]) for idi, lst in enumerate(lsta)]


def list_fac(lsta, fac):
    """Multiply list of arrays by constant factor.

    Parameters
    ----------
    lsta : list
        first list
    fac : float
        multiplication factor

    Returns
    -------
    lst : lst
        result
    """
    return [np.multiply(lsta[idi], fac) for idi, lst in enumerate(lsta)]


def list_min(lsta):
    """Find minimum in list of arrays.

    Parameters
    ----------
    lsta : list
        first list

    Returns
    -------
    idy : integer
        position of minimum
    idx : integer
        position of minimum
    """
    mins = [np.min(lst) for lst in lsta]
    idy = np.argmin(mins)
    idx = np.argmin(lsta[idy])
    return idy, idx


def is_bounded(array, vmin, vmax):
    """Check that array elements are between bounds.

    Parameters
    ----------
    array : array
        array
    vmin : float
        lower bound
    vmax : float
        upper bound

    Returns
    -------
    res : Boolean array
        elements between bounds
    """
    array = np.array(array)
    res = (array >= vmin) & (array <= vmax)
    return res


def is_intersecting(veca, vecb):
    """Check if two inputs intersect, i.e. contain identical elements.

    Parameters
    ----------
    veca : array or list or set
        set one
    vecb : array or list or set
        set two

    Returns
    -------
    out : Boolean
        two inputs intersect or not
    """
    out = len(set(veca).intersection(vecb)) > 0
    return out


def count_frequencies(tab):
    """Count numbers of occurences of elements in data frame columns.

    Parameters
    ----------
    tab : data frame
        original data

    Returns
    -------
    levelz : list of arrays
        unique elements per data frame column
    freqz : list of arrays
        frequency of elements
    """
    columns = tab.columns
    levelz = [np.unique(tab[col]) for col in columns]
    freqz = [[sum(tab[col] == val) for val in lev]
             for col, lev in zip(columns, levelz)]
    return levelz, freqz


def sample_categories_heuristic(tab, fraction, pmax=0.9):
    """Heuristically select stratified sample of data frame such that all
    values are represented with approximately equal/uniform probability.

    Parameters
    ----------
    tab : data frame
        original data
    fraction : float
        fraction of data frame rows to sample
    pmax : float
        upper limit for fraction of elements to be selected

    Returns
    -------
    index : list
        indices of data frame rows
    """

    # get element counts
    levelz, freqz = count_frequencies(tab)

    # compute average/wanted frequencies
    wanted = [pmax*np.mean(freq) for freq in freqz]
    wanted = list_add(wanted, list_fac(freqz, 0))

    # setup empty list for current counts
    current = list_fac(freqz, 0)
    leni = int(len(tab)*fraction)
    index = []

    # while not enough samples picked...
    while len(index) < leni:

        # compute difference between current and wanted counts
        diff = list_sub(current, wanted)

        # get position of element with stronged deviation from wanted count
        idy, idx = list_min(diff)

        # get row with element of interest
        whr = tab.iloc[:, idy] == levelz[idy][idx]

        # select random row that contains element
        idn = np.random.choice(whr[whr].index, 1)[0]

        # count all elements in row and add to current
        for idy, val in enumerate(tab.loc[idn]):
            idx = np.where(levelz[idy] == val)[0][0]
            current[idy][idx] += 1

        # append selected table index
        index.append(idn)

    return index


def sample_categories_optimal(tab, fraction):
    """Select stratified sample of data frame such that all
    values are represented with approximately equal/uniform probability
    using optimization via integer linear programming (ILP).

    Parameters
    ----------
    tab : data frame
        original data
    fraction : float
        fraction of data frame rows to sample

    Returns
    -------
    index : list
        indices of data frame rows
    """

    # categories
    levelz, freqz = count_frequencies(tab)

    # array sizes
    len_i, len_j = tab.shape
    len_k = max([len(lev) for lev in levelz])

    # binary representation of table categories
    y_ijk = np.zeros((len_i, len_j, len_k), dtype=np.int)
    for idi in range(len_i):
        for idj in range(len_j):
            val = tab.iloc[idi, idj]
            idk = list(levelz[idj]).index(val)
            y_ijk[idi, idj, idk] = 1

    # desired/optimum number of occurences per class categories
    opt_j = np.array([int(fraction*len_i/len(lev)) for lev in levelz])

    # upper limits
    lim_jk = np.zeros((len_j, len_k), dtype=np.int)
    for idj, freq in enumerate(freqz):
        lgt = len(freqz[idj])
        lim_jk[idj, :lgt] = freqz[idj]

    # initialize variables
    varx = (idi for idi in range(len_i))
    varsp = ((idj, idk)
             for idj in range(len_j)
             for idk in range(len_k))
    varsm = ((idj, idk)
             for idj in range(len_j)
             for idk in range(len_k))
    xxx = pulp.LpVariable.dicts('xxx', varx, cat=pulp.LpBinary)
    ssp = pulp.LpVariable.dicts('ssp', varsp, cat=pulp.LpContinuous, lowBound=0)
    ssm = pulp.LpVariable.dicts('ssm', varsm, cat=pulp.LpContinuous, lowBound=0)

    # set objective
    # sum_jk sp_jk + sm_kj
    problem = pulp.LpProblem('unify', pulp.LpMinimize)
    problem += sum([ssp[iii] for iii in ssp]) + sum([ssm[iii] for iii in ssm])

    # set constaints
    for idj in range(len_j):
        for idk in range(len_k):
            # c_jk >= 0
            problem += pulp.lpSum(xxx[idi]*y_ijk[idi, idj, idk] for idi in range(len_i)) >= 0
            # c_jk <= lim_jk
            problem += pulp.lpSum(xxx[idi]*y_ijk[idi, idj, idk] for idi in range(len_i)) <= lim_jk[idj, idk]
            # sp_jk - sm_jk = c_jk - d_j
            problem += (ssp[idj, idk] - ssm[idj, idk]) == (pulp.lpSum(xxx[idi]*y_ijk[idi, idj, idk] for idi in range(len_i)) - opt_j[idj])

    # solve problem
    print('start')
    temp = time.time()
    problem.solve()
    dura = time.time() - temp
    print('stop', dura)

    # collect results
    obj = problem.objective.value()
    rex = np.zeros(len_i, dtype=np.int)
    resp = np.zeros((len_j, len_k))
    resm = np.zeros((len_j, len_k))
    for iii in xxx:
        rex[iii] = xxx[iii].value()
    for iii in ssp:
        resp[iii] = ssp[iii].value()
    for iii in ssm:
        resm[iii] = ssm[iii].value()

    # print results
    print('fraction', fraction, np.mean(rex))
    print('objective', obj, np.sum(np.abs(resp+resm)))

    # return indices
    index = np.where(rex)[0]

    return index


def sample_categories(tab, fraction, method='random', pmax=0.9):
    """Select random sample of input table using different sampling procedures,
    see details below.

    Parameters
    ----------
    tab : data frame
        original data
    fraction : float
        fraction of data frame rows to sample
    method : string
        | name of sampling method, one of:
        | random = random sampling
        | heuristic = heuristical uniform sampling
        | optimal = uniform samplung using integer linear programming
    pmax : float
        upper limit for fraction of elements to be selected

    Returns
    -------
    index : list
        indices of data frame rows
    """
    if method == 'heuristic':
        index = sample_categories_heuristic(tab, fraction, pmax=pmax)
    elif method == 'optimal':
        index = sample_categories_optimal(tab, fraction)
    else:
        index = sample_fraction(len(tab), fraction)
    return index


# %%########################################################################### database


def connect_database(provider):
    """Connect to database with credentials stored in .env file.

    Parameters
    ----------
    provider : string
        | database provider

    Returns
    -------
    con : connection object
        database connection object
    """

    name, host, user, word = get_envs(provider)

    con = psycopg2.connect(
        database=name,
        user=user,
        password=word,
        host=host
    )

    return con


def request(cmd, token, typ='get', data=None):
    """Send request to API.

    Parameters
    ----------
    cmd : string
        API command
    token : string
        access token
    typ : string, {'get', 'post', 'delete'}
        type of request
    data : None or dict
        post data corresponding to JSON blob

    Returns
    -------
    res : string
        content of answer to request

    Examples
    --------
    >>>
    res = request('<address>', '<token>', typ='post')
    """
    headers = {'Authorization': token}
    if typ == 'get':
        func = requests.get
    elif typ == 'delete':
        func = requests.delete
    elif typ == 'post':
        func = functools.partial(requests.post, json=data)
    else:
        return None
    try:
        res = func(cmd, headers=headers).content
    except Exception as err:
        warnings.warn('API access failed: ' + str(err))
        res = None
    return res


# %%########################################################################### ternary


def convert_bary2cart(bary, tri):
    """Convert Barycentric to Cartesian coordinates.

    Parameters
    ----------
    bary : array
        Barycentric data
    tri : array
        positions of Barycentric triangle

    Returns
    -------
    out : array
        Cartesian coordinates
    """
    return tri.dot(bary)


def convert_cart2bary(cart, tri):
    """Convert Cartesian to Barycentric coordinates.

    Parameters
    ----------
    cart : array
        Cartesian data
    tri : array
        positions of Barycentric triangle

    Returns
    -------
    out : array
        Barycentric coordinates
    """
    xx0, yy0 = cart
    xx1, xx2, xx3 = tri[0]
    yy1, yy2, yy3 = tri[1]
    l1_up = (yy2-yy3)*(xx0-xx3) + (xx3-xx2)*(yy0-yy3)
    l1_lo = (yy2-yy3)*(xx1-xx3) + (xx3-xx2)*(yy1-yy3)
    l2_up = (yy3-yy1)*(xx0-xx3) + (xx1-xx3)*(yy0-yy3)
    l2_lo = (yy2-yy3)*(xx1-xx3) + (xx3-xx2)*(yy1-yy3)
    ll1 = l1_up/l1_lo
    ll2 = l2_up/l2_lo
    ll3 = 1 - ll1 - ll2
    return ll1, ll2, ll3


def convert_small2large(small, tlarge, tsmall):
    """Convert coordinates in small to large Barycentric triangle.

    Parameters
    ----------
    small : array
        small Barycentric data
    tlarge : array
        large Barycentric triangle
    tsmall : array
        small Barycentric triangle

    Returns
    -------
    out : array
        large Barycentric coordinates
    """
    cart = convert_bary2cart(small, tsmall)
    large = convert_cart2bary(cart, tlarge)
    return large


def convert_large2small(large, tlarge, tsmall):
    """Convert coordinates in large to small Barycentric triangle.

    Parameters
    ----------
    large : array
        large Barycentric data
    tlarge : array
        large Barycentric triangle
    tsmall : array
        small Barycentric triangle

    Returns
    -------
    out : array
        small Barycentric coordinates
    """
    cart = convert_bary2cart(large, tlarge)
    small = convert_cart2bary(cart, tsmall)
    return small


def make_triangles():
    """Generate large unit triangle (a/b/c) and smaller triangle
    (A/B/C).

    Parameters
    ----------
    None : None
        None

    Returns
    -------
    tlarge : array
        large Barycentric triangle
    tsmall : array
        small Barycentric triangle
    """
    tlarge = np.transpose(np.array([[0, 0], [1, 0], [0, 1]]))
    centers = pd.DataFrame([[15, 5, 80],
                            [25, 25, 50],
                            [75, 15, 15]],
                           columns=['a', 'b', 'c'],
                           index=['A', 'B', 'C'])
    tsmall = convert_bary2cart(np.transpose(centers)/100, tlarge)
    return tlarge, tsmall


def convert_classes2percentages(classes):
    """Convenience function to convert classes (A/B/C) to percentages
    (a/b/c).

    Parameters
    ----------
    classes : array
        classes, must be given as [0.1, 0.2, 0.7] and sum to one

    Returns
    -------
    out : array
        percentages
    """
    tlarge, tsmall = make_triangles()
    percentages = convert_small2large(classes, tlarge, tsmall)
    return percentages


def convert_percentages2classes(percentages):
    """Convenience function to convert percentages (a/b/c) to classes
    (A/B/C).

    Parameters
    ----------
    percentages : array
        percentages, must be given as [0.1, 0.2, 0.7] and sum to one

    Returns
    -------
    out : array
        classes
    """
    tlarge, tsmall = make_triangles()
    classes = convert_large2small(percentages, tlarge, tsmall)
    return classes


# %%########################################################################### end file
