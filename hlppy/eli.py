"""Define EliTech functions"""

# %%########################################################################### import modules

import datetime
import inspect
import os
import time

import elitech
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%########################################################################### define functions


def download_data(path):
    """Read data from EliTech EC5 USB temperature logger.

    https://pypi.org/project/elitech-datareader/
    
    pip install elitech-datareader

    Parameters
    ----------
    path : string
        path to store intermediate results

    Returns
    -------
    None : None
        None
    """

    device = elitech.Device('/dev/ttyUSB0')

    devinfo = device.get_devinfo()

    print(inspect.getmembers(devinfo))

    body = device.get_data()

    tab = pd.DataFrame(body, columns=['count', 'timestamp', 'temperature'])

    print(tab.shape)

    temp = time.strftime('%Y%m%d', time.localtime())

    file = path+temp+'.csv'

    if not os.path.isfile(file):
        print(file)
        tab.to_csv(file)

    return None


def concatenate_data(path):
    tab = pd.concat([pd.read_csv(os.path.join(path, file)) for file in os.listdir(path)]).drop_duplicates()
    tab['timestamp'] = pd.to_datetime(tab['timestamp'])
    tab['day'] = tab['timestamp'].dt.strftime('%Y%m%d')
    tab['iday'] = pd.Categorical(tab['day']).codes
    tab['daytime'] = tab['timestamp'].dt.hour + tab['timestamp'].dt.minute/60.0
    tab['bin'] = (tab['daytime']*4).map(int)
    return tab


def plot_data(tab):
    fig, axis = plt.subplots(figsize=(16, 9), nrows=2)
    # raw time
    axia = axis[0]
    axia.plot(tab['timestamp'], tab['temperature'], color='black')
    axia.plot(tab['timestamp'], tab['temperature']*0+20, color='gray')
    axia.set_xlabel('datetime')
    axia.set_ylabel('Temperature [°C]')
    # per day
    axia = axis[1]
    vec = np.unique(tab['bin'])/4
    groups = tab.groupby('iday')
    bins = tab.groupby('bin')
    avg = bins.mean()['temperature']
    std = bins.std()['temperature']
    axia.fill_between(vec, avg-std, avg+std, color='black', lw=0, alpha=0.2)
    for idg, group in groups:
        color = plt.cm.jet(1.0*idg/max(tab['iday']))
        axia.plot(group['daytime'], group['temperature'], color=color, alpha=0.6)
    axia.plot(vec, avg, color='black', lw=4)
    axia.set_xlabel('daytime')
    axia.set_ylabel('Temperature [°C]')
    return None


def set_parameters():
    device = elitech.Device('/dev/ttyUSB0')
    devinfo = device.get_devinfo()
    param_put = devinfo.to_param_put()
    param_put.current = datetime.datetime.now()
    param_put.rec_interval = datetime.time(0, 10)
    param_put.stop_button = elitech.StopButton.ENABLE
    param_put_res = device.update(param_put)
    return param_put_res


# %%########################################################################### end file
