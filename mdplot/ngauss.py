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
        try:
            msd = H5._v_children['MSD']
            mqd = H5._v_children['MQD']
            # discard time zero
            msd = msd[:, 1:, :]
            mqd = mqd[:, 1:, :]

            if args.unordered:
                x = msd[:, :, 0]
                y1 = msd[:, :, 1]
                y2 = mqd[:, :, 1]
            else:
                msd.shape = -1, msd.shape[-1]
                mqd.shape = -1, mqd.shape[-1]
                # time-order correlation function samples
                time_order = msd[:, 0].argsort()
                x = msd[time_order, 0]
                y1 = msd[time_order, 1]
                y2 = mqd[time_order, 1]

            y = 3 * y2 / (5 * y1 * y1) - 1

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

    plot.setp(ax, xlabel=args.xlabel or r'$t^*$')
    plot.setp(ax, ylabel=args.ylabel or r'$\frac{3}{5} \frac{\langle\delta r(t^*)^4 \rangle}{\langle\delta r(t^*)^2 \rangle^2} - 1$')

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('ngauss', help='non-Gaussian parameter')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlation file')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--unordered', action='store_true', help='disable block time ordering')

