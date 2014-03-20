#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.distutils.core import setup, Extension
from subprocess import check_output

def parse_git_version():
    version = 'unkown'
    try:
        version = check_output(['git', 'describe', '--always']).strip() or version
    except OSError:
        pass
    return version

module1 = Extension('h5md._plot.ext',
                     sources = ['h5md/_plot/c/ext.cpp', 'h5md/_plot/c/ssf.cpp'],
                     extra_compile_args = ['-O3', '-fopenmp', '-mtune=native', '-Wall'],
                     extra_link_args = ['-fopenmp'])

setup(name = 'h5md-tools',
      version = parse_git_version(),
      description = 'Toolset for H5MD files',
      author = ('Felix Höfling'),
      author_email = ('hoefling@mf.mpg.de'),
      packages = ['h5md', 'h5md._plot'],
      scripts = ['bin/h5md'],
      package_data={'h5md._plot': ['gpu/ssf_kernel.cu']},
      ext_modules = [module1,],
      license = 'GPL'
      )
