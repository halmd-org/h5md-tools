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
Plot correlation functions
"""
def plot(args):
    from matplotlib import pyplot as plot

    ax = plot.axes()
    title = None

    # plot line for zero crossings
    if not args.axes in ('ylog', 'loglog'):
        ax.axhline(y=0, color='black')

    ci = 0
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        try:
            data = H5._v_children[args.type]
            # merge block levels, discarding time zero
            tcf = data[:, 1:, :]
            if args.unordered:
                x, y, yerr = tcf[:, :, 0], tcf[:, :, 1], tcf[:, :, 2]
            else:
                tcf.shape = -1, tcf.shape[-1]
                # prepend time zero from lowest block
                tcf = concatenate((data[0, 0, :].reshape(1, tcf.shape[-1]), tcf))
                # time-order correlation function samples
                time_order = tcf[:, 0].argsort()
                x, y, yerr = tcf[time_order, 0], tcf[time_order, 1], tcf[time_order, 2]

            if args.label:
                label = args.label % mdplot.label.attributes(H5.param)
            else:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = r'{\small %s}' % basen.replace('_', r'\_')
            if args.title:
                title = args.title % mdplot.label.attributes(H5.param)

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % fn)

        finally:
            f.close()

        if args.xaxis:
            # limit data points to given x-axis range
            i = where((x >= args.xaxis[0]) & (x <= args.xaxis[1]))
            x, y, yerr = x[i], y[i], yerr[i]
        if args.yaxis:
            # limit data points to given y-axis range
            i = where((y >= args.yaxis[0]) & (x <= args.yaxis[1]))
            x, y, yerr = x[i], y[i], yerr[i]

        if args.normalize:
            y, yerr = (y / y[0]), (yerr / y[0])

        if not len(x) or not len(y):
            raise SystemExit('empty plot range')


        # cycle plot color
        c = args.colors[ci % len(args.colors)]
        ci += 1
        if args.unordered:
            # plot start point of each block
            ax.plot(x[:, 0], y[:, 0], '+', color=c, ms=10, alpha=0.5, label=label)
            for (i, (bx, by, byerr)) in enumerate(zip(x, y, yerr)):
                # plot single block
                ax.plot(bx, by, marker=(',', '3')[i % 2], color=c, lw=0.1, ms=3)
        else:
            ax.errorbar(x, y, yerr=yerr[0], color=c, label=label)

    # optionally plot with logarithmic scale(s)
    if args.axes == 'xlog':
        ax.set_xscale('log')
    if args.axes == 'ylog':
        ax.set_yscale('log')
    if args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    l = ax.legend(loc=args.legend, labelsep=0.01, pad=0.1, axespad=0.025)
    l.legendPatch.set_alpha(0.7)

    if not title is None:
        plot.title(title)

    plot.axis('tight')
    plot.xlabel(r'$\tau$')
    ylabel = {
        'MSD': r'$\langle(r(t+\tau)-r(t))^2\rangle$',
        'MQD': r'$\langle(r(t+\tau)-r(t))^4\rangle$',
        'VAC': r'$\langle v(t+\tau)v(t)\rangle$',
    }
    plot.ylabel(ylabel[args.type])

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('corr', help='correlation functions')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlations file')
    parser.add_argument('--type', required=True, choices=['MSD', 'MQD', 'VAC'], help='correlation function')
    parser.add_argument('--xaxis', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--yaxis', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--unordered', action='store_true', help='disable block time ordering')
    parser.add_argument('--normalize', action='store_true', help='normalize function')

