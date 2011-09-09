#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.distutils.core import setup, Extension
from subprocess import Popen, PIPE

def parse_git_version():
    version = 'unkown'
    try:
        version = Popen(['git', 'describe', '--always'], stdout=PIPE).communicate()[0] or version
    except OSError:
        pass
    return version

module1 = Extension('mdplot.ext',
                     sources = ['mdplot/c/ext.cpp', 'mdplot/c/ssf.cpp'],
                     extra_compile_args = ['-O3', '-fopenmp', '-mtune=native', '-Wall'],
                     extra_link_args = ['-fopenmp'])

module2 = Extension('h5md._plot.ext',
                     sources = ['h5md/_plot/c/ext.cpp', 'h5md/_plot/c/ssf.cpp'],
                     extra_compile_args = ['-O3', '-fopenmp', '-mtune=native', '-Wall'],
                     extra_link_args = ['-fopenmp'])

setup(name = 'h5md',
      version = parse_git_version(),
      description = 'Toolset for H5MD files',
      author = ('Peter Colberg', 'Felix Höfling'),
      author_email = ('peter.colberg@utoronto.ca', 'hoefling@mf.mpg.de'),
      packages = ['h5md', 'h5md._plot', 'mdplot'],
      scripts = ['bin/h5md', 'bin/mdplot', 'bin/compute_msv'],
      package_data={'mdplot': ['gpu/ssf_kernel.cu'], 'h5md._plot': ['gpu/ssf_kernel.cu']},
      ext_modules = [module1, module2],
      license = 'GPL'
      )
