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
from numpy import *
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
        nq = H5.param.correlation._v_attrs.q_values
        # merge block levels, discarding time zero
        data = H5._v_children[args.type][:, :, 1:, :]
        data.shape = nq, -1, data.shape[-1]
        # F(q, 0) at lowest block level
        norm = H5._v_children[args.type][:, 0, 0, 2]

    except tables.exceptions.NoSuchNodeError:
        raise SystemExit('missing simulation data in file: %s' % args.input)

    finally:
        f.close()

    ax = plt.axes()
    # plot zero line
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_color_cycle(args.colors)

    # q-vector magnitudes
    q_values = sorted(unique(data[:, :, 0]))
    if args.q_values:
        # generate intervals with q-vectors in interval centers
        q_bins = (q_values + append((0, ), q_values[:-1])) / 2
        # choose nearest neighbour q-vectors
        q_values = choose(digitize(args.q_values, q_bins) - 1, q_values)

    for q in q_values:
        # select blocks with matching q-vector
        d = data.compress(data[:, 0, 0] == q, axis=0)
        # select norms with matching q-vector
        n = norm.compress(data[:, 0, 0] == q, axis=0)
        # time-order correlation function samples
        time_order = d[0, :, 1].argsort()
        x, y, yerr = d[0, time_order, 1], d[0, time_order, 2], d[0, time_order, 3]

        if args.normalize:
            # normalize with F(q, 0)
            y = y / n
            yerr = yerr / n

        if not len(x):
            raise SystemExit('empty plot range')

        ax.errorbar(x, y, yerr=yerr, label=r'$q = %.2f$' % q)

    ax.set_xscale('log')
    l = ax.legend(loc=args.legend)
    l.legendPatch.set_alpha(0.7)

    plt.axis('tight')
    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    ylabel = {
        'ISF': r'$F(q, \tau)$',
        'SISF': r'$F_s(q, \tau)$',
    }
    plt.xlabel(args.xlabel or r'$\tau$')
    plt.ylabel(args.ylabel or ylabel[args.type])

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('isf', help='intermediate scattering function')
    parser.add_argument('input', metavar='INPUT', help='HDF5 correlations file')
    parser.add_argument('--type', required=True, choices=['ISF', 'SISF'], help='correlation function')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--q-values', type=float, nargs='+', help='q-vector magnitude(s)')
    parser.add_argument('--normalize', action='store_true', help='normalize function')

