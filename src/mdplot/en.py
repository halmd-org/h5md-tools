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
Plot mean total energy per particle
"""
def plot(args):
    from matplotlib import pyplot as plt

    ylabel = {
        'ETOT':  r'$\langle E\rangle / \epsilon$',
        'EPOT':  r'$\langle U\rangle / \epsilon$',
        'EKIN':  r'$\langle T\rangle / \epsilon$',
        'PRESS': r'$\langle P\rangle$',
        'TEMP':  r'$\langle T\rangle$',
        'VCM':   r'$\vert\langle \textbf{v}^*\rangle\vert$',
    }
    mlabel = {
        'ETOT':  r'$\langle\langle E\rangle\rangle_{t^*}$',
        'EPOT':  r'$\langle\langle U\rangle\rangle_{t^*}$',
        'EKIN':  r'$\langle\langle T\rangle\rangle_{t^*}$',
        'PRESS': r'$\langle\langle P\rangle\rangle_{t^*}$',
        'TEMP':  r'$\langle\langle T\rangle\rangle_{t^*}$',
        'VCM':   r'$\langle\vert\langle \textbf{v}^*\rangle\vert\rangle_{t^*}$',
    }

    ax = plt.axes()

    ci = 0
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % args.input)

        H5 = f.root
        try:
            data = H5._v_children[args.type]
            x = data[:, 0]
            if args.type == 'VCM':
                # positional coordinates dimension
                dim = H5.param.mdsim._v_attrs.dimension
                # calculate center of velocity magnitude
                if dim == 3:
                    y = numpy.sqrt(data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2] + data[:, 3] * data[:, 3])
                else:
                    y = numpy.sqrt(data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2])
            else:
                y = data[:, 1]

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % args.input)

        finally:
            f.close()

        if args.xaxis:
            # limit data points to given x-axis range
            i = numpy.where((x >= args.xaxis[0]) & (x <= args.xaxis[1]))
            x, y = x[i], y[i]

        if not len(x):
            raise SystemExit('empty plot range')

        # cycle plot color
        c = args.colors[ci % len(args.colors)]
        ci += 1

        ax.plot(x, y, color=c)

        if args.mean:
            m, s = numpy.mean(y), numpy.std(y)
            # plot standard deviation
            ax.axhspan(m - s, m + s, facecolor=c, edgecolor=c, alpha=0.1)
            # plot mean
            ax.axhline(m, linestyle='--', color=c, alpha=0.5)
            ax.text(1.01 * x.max() - 0.01 * x.min(), m, mlabel[args.type], verticalalignment='center', horizontalalignment='left')

    major_formatter = ticker.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    plt.xlabel(r'$t^*$')
    plt.ylabel(ylabel[args.type])

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('en', help='thermal equilibrium properties')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 energy file')
    parser.add_argument('--type', required=True, choices=['ETOT', 'EPOT', 'EKIN', 'PRESS', 'TEMP', 'VCM'], help='thermal equilibrium property')
    parser.add_argument('--xaxis', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--mean', action='store_true', help='plot mean and standard deviation')

