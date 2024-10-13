#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from subprocess import check_output
import numpy

def parse_git_version():
    version = 'unkown'
    try:
        version = check_output(['git', 'describe', '--always']).decode('utf-8').strip() or version
        # make version string compliant with Python restrictions for non-release Git commits
        version = version.replace('-', '+dirty', 1)     # replace first '-' by '+dirty'
        version = version.replace('-', '.')             # replace remaining '-' by '.'
    except OSError:
        pass
    return version

module1 = Extension('h5mdtools._plot.ext',
                     sources = ['h5mdtools/_plot/c/ext.cpp', 'h5mdtools/_plot/c/ssf.cpp'],
                     include_dirs = [numpy.get_include()],
                     extra_compile_args = ['-O3', '-fopenmp', '-mtune=native', '-Wall'],
                     extra_link_args = ['-fopenmp'])

setup(name = 'h5md-tools',
      version = parse_git_version(),
      description = 'Toolset for H5MD files',
      author = ('Felix Höfling'),
      author_email = ('f.hoefling@fu-berlin.de'),
      packages = ['h5mdtools', 'h5mdtools._plot'],
      console_scripts = ['bin/h5md'],
      package_data={'h5mdtools._plot': ['gpu/ssf_kernel.cu']},
      ext_modules = [module1,],
      license = 'GPL'
      )
