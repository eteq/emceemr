#!/usr/bin/env python

import re
from setuptools import setup

# Hackishly synchronize the version. Inspired by dfm's trangle.py
version = re.findall(r'__version__ = \'(.*?)\'', open('emceemr/__init__.py').read())[0]

setup(
    name='emceemr',
    author='Erik Tollerud',
    author_email='erik.tollerud@gmail.com',
    packages=['emceemr'],
    description='An package providing a higher-level modeling approach to '
                'MCMC/Bayesian inferential fitters.',
    long_description=open('README.rst').read(),
    version=version,
    package_requires=['numpy', 'scipy', 'astropy'],
    python_requires='>=3.7',  #data classes
    install_requires=['setuptools']
)
