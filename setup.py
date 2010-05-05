#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from subprocess import Popen, PIPE
import os

def parse_git_version():
    version = 'unkown'
    try:
        version = Popen(['git', 'describe', '--always'], stdout=PIPE).communicate()[0] or version
    except OSError:
        pass
    return version

incdir = os.path.expanduser('~/usr/lib/python2.6/site-packages/numpy-1.4.0-py2.6-linux-x86_64.egg/numpy/core/include')

module1 = Extension('mdplot.ext',
                     sources = ['mdplot/c/ext.cpp', 'mdplot/c/ssf.cpp'],
                     include_dirs = [incdir],
                     extra_compile_args = ['-O3', '-mtune=native', '-Wall'])

setup(name = 'mdplot',
      version = parse_git_version(),
      description = 'Molecular Dynamics simulation plotter',
      author = ('Peter Colberg', 'Felix Höfling'),
      author_email = ('peter.colberg@physik.uni-muenchen.de', 'hoefling@mf.mpg.de'),
      packages = ['mdplot'],
      scripts = ['bin/mdplot'],
      license = 'GPL',
      ext_modules = [module1]
      )

