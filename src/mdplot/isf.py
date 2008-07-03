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
from matplotlib import ticker
import numpy
import sys
import tables


"""
Plot intermediate scattering function
"""
def plot(args):
    from matplotlib import pyplot as plt

    try:
        f = tables.openFile(args.input, mode='r')
    except IOError:
        raise SystemExit('failed to open HDF5 file: %s' % args.input)

    H5 = f.root
    try:
        # number of q-values
        q_values = H5.param.correlation._v_attrs.q_values
        # merge block levels
        data = numpy.reshape(H5._v_children[args.type][:, :, 1:, :], (q_values, -1, 4))
        # F(q, 0) at lowest block level
        norm = H5._v_children[args.type][:, 0, 0, 2]

    except tables.exceptions.NoSuchNodeError:
        raise SystemExit('missing simulation data in file: %s' % args.input)

    finally:
        f.close()

    ax = plt.axes()
    ax.set_color_cycle(['m', 'b', 'c', 'g', 'r'])

    for (d, n) in zip(data, norm):
        q = d[0, 0]
        # time-order correlation function samples
        i = d[:, 1].argsort()
        x, y, yerr = d[i, 1], d[i, 2], d[i, 3]

        if args.normalize:
            # normalize with F(q, 0)
            y = y / n
            yerr = yerr / n

        if args.xaxis:
            # limit data points to given x-axis range
            i = numpy.where((x >= args.xaxis[0]) & (x <= args.xaxis[1]))
            x, y, yerr = x[i], y[i], yerr[i]

        if not len(x):
            raise SystemExit('empty plot range')

        ax.errorbar(x, y, yerr=yerr, label=r'$q = %.2f$' % q)

    ax.set_xscale('log')
    ax.legend()

    ylabel = {
        'ISF': r'$F(q, \tau)$',
        'SISF': r'$F_s(q, \tau)$',
    }
    plt.axis('tight')
    plt.xlabel(r'$\tau$')
    plt.ylabel(ylabel[args.type])

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('isf', help='intermediate scattering function')
    parser.add_argument('input', metavar='INPUT', help='HDF5 correlations file')
    parser.add_argument('--type', required=True, choices=['ISF', 'SISF'], help='correlation function')
    parser.add_argument('--xaxis', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--normalize', action='store_true', help='normalize function')

