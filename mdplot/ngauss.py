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
import mdplot.label


"""
Plot non-Gaussian parameter
"""
def plot(args):
    from matplotlib import pyplot as plot

    ax = plot.axes()
    label = None
    title = None

    for i, fn in enumerate(args.input):
        # cycle plot color
        c = args.colors[i % len(args.colors)]

        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        dimension = H5.param.mdsim._v_attrs.dimension
        try:
            msd = H5._v_children['MSD']
            mqd = H5._v_children['MQD']
            # discard time zero
            msd = msd[:, 1:, :]
            mqd = mqd[:, 1:, :]

            x = msd[:, :, 0]
            if args.type == 'BURNETT':
                h = x[:, 1:] - x[:, :-1]
                x = (x[:, 1:] + x[:, :-1]) / 2
                # Burnett coefficient from central difference
                y = dimension * mqd[:, :, 1] / (2 + dimension) - pow(msd[:, :, 1], 2)
                y = diff(y, axis=1, n=1) / (24 * h)
            else:
                # non-Gaussian parameter
                y = dimension * mqd[:, :, 1] / ((2 + dimension) * pow(msd[:, :, 1], 2)) - 1

            if not args.unordered:
                x.shape = -1
                y.shape = -1
                # time-order correlation function samples
                time_order = x.argsort()
                x, y = x[time_order], y[time_order]

            if args.label:
                label = args.label[i % len(args.label)] % mdplot.label.attributes(H5.param)
            elif args.legend or not args.small:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')
            if args.title:
                title = args.title % mdplot.label.attributes(H5.param)

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % fn)

        finally:
            f.close()

        if not len(x) or not len(y):
            raise SystemExit('empty plot range')

        if args.unordered:
            # plot start point of each block
            ax.plot(x[:, 0], y[:, 0], '+', color=c, ms=10, alpha=0.5, label=label)
            # plot separate curve for each block
            for (i, (xi, yi)) in enumerate(zip(x, y)):
                ax.plot(xi, yi, marker=(',', '3')[i % 2], color=c, lw=0.2, ms=3)

        else:
            ax.plot(x, y, color=c, label=label)

    if args.axes == 'xlog':
        ax.set_xscale('log')
    elif args.axes == 'ylog':
        ax.set_yscale('log')
    elif args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    if not title is None:
        plot.title(title)

    ax.axis('tight')
    if args.xlim:
        plot.setp(ax, xlim=args.xlim)
    if args.ylim:
        plot.setp(ax, ylim=args.ylim)

    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    def frac(a, b):
        d = gcd(a, b)
        if b == d:
            return '%d' % (a / d)
        else:
            return r'\frac{%d}{%d}' % (a / d, b / d)

    if args.type == 'BURNETT':
        ylabel = r'$\frac{1}{4!}\frac{d}{dt}(%s\langle\delta r(t^*)^4 \rangle - \langle\delta r(t^*)^2 \rangle^2)$' % frac(dimension, dimension + 2)
    else:
        ylabel = r'$%s\frac{\langle\delta r(t^*)^4 \rangle}{\langle\delta r(t^*)^2 \rangle^2} - 1$' % frac(dimension, dimension + 2)

    plot.setp(ax, xlabel=args.xlabel or r'$t^*$')
    plot.setp(ax, ylabel=args.ylabel or ylabel)

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('ngauss', help='non-Gaussian parameter')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlation file')
    parser.add_argument('--type', choices=['NGAUSS', 'BURNETT'], help='correlation function')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--unordered', action='store_true', help='disable block time ordering')

