#!/usr/bin/env python
"""Copy music albums."""

from distutils.dir_util import copy_tree
import os
import shutil
import sys

import hlppy
import hlppy.help
import hlppy.music


if __name__ == '__main__':

    # get script arguments
    ARGS = sys.argv

    # set parameters
    write = '/media/Volume/HandyMusik'
    read = '/media/Volume/David/Music'
    handy = '/run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C006%5D/Card'
    space = 50*1000

    # copy albums
    count, size = hlppy.music.copy_music(write, read, space)
    print(count, size)
    
    # nope!
    # folders = hlppy.help.list_files(write)
    # lenf = len(folders)
    # for idf, folder in enumerate(folders):
    #     print(idf, lenf)
    #     name = os.path.basename(folder)        
    #     letter = os.path.join(handy, name[:1])+'/'
    #     new = os.path.join(letter, name)
    #     hlppy.help.make_folder(letter)
    #     if os.path.isdir(new):
    #         continue
    #     #copy_tree(folder, letter)
    #     shutil.copytree(folder, new)
        

    # nope!
    # copy to card
    # sudo apt-get install gvfs-bin
    # name = os.path.basename(write)
    # os.system('gvfs-copy -p %s "mtp://[usb:001,004]/Card/%s"' % (write, name))

    # yes!
    # https://askubuntu.com/questions/343502/how-to-rsync-to-android
    # rsync --verbose --progress --ignore-existing --omit-dir-times --no-perms --recursive --inplace /media/Volume/HandyMusik/ /run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C006%5D/Card/HandyMusik/

    # add linebreak
    print('')
