#
# mdplot - Molecular Dynamics simulation plotter
#
# Copyright (C) 2008  Peter Colberg <peter.colberg@physik.uni-muenchen.de>
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

import os, os.path
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy
import sys
import tables


"""
Plot correlation functions
"""
def plot(args):
    try:
        f = tables.openFile(args.input, mode='r')
    except IOError:
        raise SystemExit('failed to open HDF5 file: %s' % args.input)

    H5 = f.root
    try:
        # merge block levels
        data = numpy.reshape(H5._v_children[args.type], (-1, 3))
        # time-order correlation function samples
        ord = data[:, 0].argsort()
        x, y, yerr = data[ord, 0], data[ord, 1], data[ord, 2]

    except tables.exceptions.NoSuchNodeError:
        raise SystemExit('missing simulation data in file: %s' % args.input)

    finally:
        f.close()

    if args.xaxis:
        # limit data points to given x-axis range
        i = numpy.where((x >= args.xaxis[0]) & (x <= args.xaxis[1]))
        x, y, yerr = x[i], y[i], yerr[i]

    if not len(x):
        raise SystemExit('empty plot range')

    ylabel = {
        'MSD': r'$\langle(r(t+\delta t)-r(t))^2\rangle$',
        'MQD': r'$\langle(r(t+\delta t)-r(t))^4\rangle$',
        'VAC': r'$\langle v(t+\delta t)v(t)\rangle$',
    }

    ax = plt.axes()
    ax.errorbar(x, y, yerr=yerr, color='m')
    ax.set_xscale('log')
    if args.type in ('MSD', 'MQD'):
        ax.set_yscale('log')

    plt.axis('tight')
    plt.xlabel(r'$\delta t$')
    plt.ylabel(ylabel[args.type])
    plt.savefig(args.output)


def add_parser(subparsers):
    parser = subparsers.add_parser(command, help='correlation functions')
    parser.add_argument('input', metavar='INPUT', help='HDF5 correlations file')
    parser.add_argument('--type', required=True, choices=['MSD', 'MQD', 'VAC'], help='correlation function')
    parser.add_argument('--xaxis', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')

command = 'corr'

