"""
Welcome everyone!
=================

Welcome everyone to this simple gallery example. See details:
https://sphinx-gallery.github.io/stable/index.html
"""

# %%########################################################################### import modules

import os
import pkg_resources

import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt

# %%########################################################################### plot image

# get image from package
loca = os.path.join('data', 'logo.png')
name = pkg_resources.resource_filename('hlppy', loca)

# plot image
fig, axis = plt.subplots(figsize=(9, 6))
imag = matplotlib.image.imread(name)
axis.imshow(imag, interpolation='bicubic')
axis.set_axis_off()
plt.tight_layout(pad=0)

# say welcome
print('Welcome everyone!')

# %%########################################################################### end module
