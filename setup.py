# import modules
from setuptools import setup, find_packages


# set package parameters
# run 'make install' to install the package
setup(name='hlppy',
      description='Code snippets and help functions',
      author='David Breuer <info@info.com>',
      version='v0.0.0',
      url='http://www.info.com/',
      packages=find_packages(),
      include_package_data=True,
      scripts=['bin/run_eli.py',
               'bin/run_music.py'],
      python_requires='==3.*',
      install_requires=[]
)
