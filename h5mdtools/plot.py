#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# plot - plotting of H5MD data
#
# Copyright © 2008-2011  Peter Colberg, Felix Höfling
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
#

import argparse
from ._plot import *
from . import _plot
import sys

def main(args):
    # load packages not before invocation of plot package
    from matplotlib import rc
    import matplotlib

    # set matplotlib defaults
    rc('font', family='serif', serif=['Computer Modern Roman'])
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{underscore}'])
    rc('legend', borderpad=0.2, labelspacing=0.01, borderaxespad=0.025, numpoints=1, fontsize=12)
    if args.a4:
        rc('figure', figsize=(11.7, 8.3))
    elif args.small:
        rc('figure', figsize=(4, 3))
        rc('font', size=8)
        rc('legend', fontsize=8)

    # set matplotlib backend
    if args.output is None:
        pass
#        matplotlib.use('GTKAgg')
    else:
        matplotlib.use('Agg')

    # execute plot command
    try:
        plots[args.plot_command].plot(args)

    except SystemExit as status:
        exit('ERROR: %s' % status)

def add_parser(subparsers):
    parser = subparsers.add_parser('plot', help='plotting H5MD data')
    parser.add_argument('--output', metavar='FILE', help='output filename')
    parser.add_argument('--dump', metavar='FILE', help='dump data to file')
    parser.add_argument('--a4', action="store_true", help='DIN A4 format')
    parser.add_argument('--small', action="store_true", help='small figure size')
    parser.add_argument('--dpi', type=float, help='resolution in dots per inch')
    parser.add_argument('--colors', nargs='+', help='plot colors')
    parser.add_argument('--label', nargs='+', help='legend label format')
    parser.add_argument('--legend', help='legend placement')
    parser.add_argument('--xlabel', help='x-axis label')
    parser.add_argument('--ylabel', help='y-axis label')
    parser.add_argument('--title', help='plot title format')
    parser.set_defaults(
            dpi=300,
            # http://blogs.mathworks.com/pick/2008/08/15/colors-for-your-multi-line-plots/
            colors=[
                (0.00, 0.00, 1.00),
                (0.00, 0.50, 0.00),
                (1.00, 0.00, 0.00),
                (0.00, 0.75, 0.75),
                (0.75, 0.00, 0.75),
                (0.75, 0.75, 0.00),
                (0.25, 0.25, 0.25),
                (0.75, 0.25, 0.25),
                (0.95, 0.95, 0.00),
                (0.25, 0.25, 0.75),
                (0.75, 0.75, 0.75),
                (0.00, 1.00, 0.00),
                (0.76, 0.57, 0.17),
                (0.54, 0.63, 0.22),
                (0.34, 0.57, 0.92),
                (1.00, 0.10, 0.60),
                (0.88, 0.75, 0.73),
                (0.10, 0.49, 0.47),
                (0.66, 0.34, 0.65),
                (0.99, 0.41, 0.23),
            ],
            )

    subparsers = parser.add_subparsers(dest='plot_command', help='available plot modules')
    for plot in list(plots.values()):
        plot.add_parser(subparsers)

plots = dict([(m, sys.modules['h5mdtools._plot.%s' % m]) for m in _plot.__all__])
