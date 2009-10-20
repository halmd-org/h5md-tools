#!/usr/bin/env python2.5
from distutils.core import setup
from subprocess import Popen, PIPE

def parse_git_version():
    version = 'unkown'
    try:
        version = Popen(['git', 'describe', '--always'], stdout=PIPE).communicate()[0] or version
    except OSError:
        pass
    return version

setup(name = 'mdplot',
      version = parse_git_version(),
      description = 'Molecular Dynamics simulation plotter',
      author = 'Peter Colberg',
      author_email = 'peter.colberg@physik.uni-muenchen.de',
      packages = ['mdplot'],
      scripts = ['bin/mdplot'],
      license = 'GPL',
      )

