"""Define EliTech functions"""

# %%########################################################################### import modules

import os
import random
import shutil

# %%########################################################################### define functions


def get_size(path):
    """Get size of folder.

    Parameters
    ----------
    path : string
        path to folder

    Returns
    -------
    size : float
        size in MB
    """
    size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            size += os.path.getsize(filepath)
    size = size/1e6
    return size


def copy_music(write, read, space, limit=200):
    """Get size of folder.

    Parameters
    ----------
    write : string
        path to write folder
    write : string
        path to write folder
    space : integer
        size of albums to copy in MB
    limit : integer
        maximum size of albums to copy in MB
        
    Returns
    -------
    count : integer
        number of copied albums
    size : float
        size of copied albums
    """
    print('Read albums')
    bands=os.listdir(read)
    bands=[b for b in bands if os.path.isdir(read+'/'+b) and b[0]!='.']
    folders=[]
    for b in bands:
        albums=os.listdir(read+'/'+b)
        albums=[a for a in albums if os.path.isdir(read+'/'+b) and a[0]!='.']
        for a in albums:
            size=get_size(read+'/'+b+'/'+a)
            if(size<limit and os.path.isdir(read+'/'+b+'/'+a)):
                folders.append(read+'/'+b+'/'+a)
    print('Copy albums')
    sizo=get_size(write)
    size=sizo
    news=[]
    count=1
    while(size<space):
        f=random.sample(folders,1)[0]
        news.append(f)
        name=f.split('/')[-2:]
        name=' - '.join(name)
        print(count, int(size), space, name)
        target=write+'/'+name
        if(not os.path.isdir(target)):
            shutil.copytree(f,target)
            size=get_size(write)
            count+=1
    return count, size


# %%########################################################################### end file
