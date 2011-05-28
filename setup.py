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

setup(name = 'mdplot',
      version = parse_git_version(),
      description = 'Molecular Dynamics simulation plotter',
      author = ('Peter Colberg', 'Felix Höfling'),
      author_email = ('peter.colberg@physik.uni-muenchen.de', 'hoefling@mf.mpg.de'),
      packages = ['mdplot'],
      scripts = ['bin/mdplot', 'bin/compute_msv', 'bin/h5md_cat'],
      package_data={'mdplot': ['gpu/ssf_kernel.cu']},
      ext_modules = [module1],
      license = 'GPL'
      )
