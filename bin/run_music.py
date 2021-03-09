#!/usr/bin/env python
"""Copy music albums."""

import os
import sys

import hlppy
import hlppy.music


if __name__ == '__main__':

    # get script arguments
    ARGS = sys.argv

    # set parameters
    write = '/media/Volume/David/2020-10-11'
    read = '/media/Volume/David/Music'
    space = 100

    # copy albums
    count, size = hlppy.music.copy_music(write, read, space)
    print(count, size)

    # nope!
    # copy to card
    # sudo apt-get install gvfs-bin
    # name = os.path.basename(write)
    # os.system('gvfs-copy -p %s "mtp://[usb:001,004]/Card/%s"' % (write, name))

    # yes!
    # https://askubuntu.com/questions/343502/how-to-rsync-to-android
    # rsync --verbose --progress --ignore-existing --omit-dir-times --no-perms --recursive --inplace /media/Volume/David/2020-10-11/ /run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C006%5D/Card/2020-10-11/

    # add linebreak
    print('')
