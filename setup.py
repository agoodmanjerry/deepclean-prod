#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) DeepClean Group (2019)
#
# This file is part of DeepClean.
# DeepClean is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepClean is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepClean. If not, see <http://www.gnu.org/licenses/>.


import pkg_resources
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = [str(r) for r in pkg_resources.parse_requirements(f)]

__version__ = '0.0.0'
    
setup(
    name='DeepClean Production',
    version=__version__,
    author='''Tri Nguyen, Michael Coughlin, Rich Ormiston, Rana Adhikari, Gabriele Vajente, Erik Katsavounidis''',
    author_email='tri.nguyen@ligo.org',
    maintainer='Tri Nguyen',
    maintainer_email='tri.nguyen@ligo.org',
    license='LICENSE',
    description='Production pipeline for DeepClean: A noise regression pipeline for GW detectors.',
    long_description=open('README.md').read(),
    packages=find_packages(),
    classifiers=(
      'Programming Language :: Python',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: End Users/Desktop',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Physics',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Operating System :: MacOS',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ),
    scripts=(
        'bin/dc-prod-train',
        'bin/dc-prod-clean',
        'bin/dc-prod-segment',
    ),
    install_requires=install_requires,
    python_requires='>=3.5',
)

