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

    if args.power:
        try:
            power_law = reshape(args.power, (-1, 4))
        except ValueError:
            raise SystemExit('power law requires 4 parameters')

    for i, fn in enumerate(args.input):
        # cycle plot color
        c = args.colors[i % len(args.colors)]

        for dset in args.type:
            try:
                f = tables.openFile(fn, mode='r')
            except IOError:
                raise SystemExit('failed to open HDF5 file: %s' % fn)

            H5 = f.root
            try:
                data = H5._v_children[dset[-3:]]
                # merge block levels, discarding time zero
                tcf = data[:, 1:, :]

                if dset == 'DIFF2MSD':
                    # calculate VACF from 2nd discrete derivative of MSD
                    h = (tcf[:, 2:, 0] - tcf[:, :-2, 0]) / 2
                    x = (tcf[:, 2:, 0] + tcf[:, :-2, 0]) / 2
                    y = 0.5 * diff(tcf[:, :, 1], axis=1, n=2) / pow(h, 2)

                    if not args.unordered:
                        x.shape = -1
                        y.shape = -1
                        # time-order correlation function samples
                        time_order = x.argsort()
                        x, y = x[time_order], y[time_order]

                    if args.normalize:
                        y0 = H5._v_children['VAC'][0, 0, 1]
                        y = y / y0

                else:
                    if args.unordered:
                        x, y, yerr = tcf[:, :, 0], tcf[:, :, 1], tcf[:, :, 2]
                    else:
                        tcf.shape = -1, tcf.shape[-1]
                        # prepend time zero from lowest block
                        tcf = concatenate((data[0, 0, :].reshape(1, tcf.shape[-1]), tcf))
                        # time-order correlation function samples
                        time_order = tcf[:, 0].argsort()
                        x, y, yerr = tcf[time_order, 0], tcf[time_order, 1], tcf[time_order, 2]

                    if args.normalize:
                        y0 = data[0, 0, 1]
                        y, yerr = (y / y0), (yerr / y0)

                if args.label:
                    label = args.label[i % len(args.label)] % mdplot.label.attributes(H5.param)
                else:
                    basen = os.path.splitext(os.path.basename(fn))[0]
                    label = r'%s:%s' % (dset, basen.replace('_', r'\_'))
                if args.title:
                    title = args.title % mdplot.label.attributes(H5.param)

            except tables.exceptions.NoSuchNodeError:
                raise SystemExit('missing simulation data in file: %s' % fn)

            finally:
                f.close()

            if args.axes in ('ylog', 'loglog'):
                # use absolute y-values with logarithmic plot (for VACF)
                y = abs(y)

            if not len(x) or not len(y):
                raise SystemExit('empty plot range')


            if args.unordered:
                # plot start point of each block
                ax.plot(x[:, 0], y[:, 0], '+', color=c, ms=10, alpha=0.5, label=label)
                # plot separate curve for each block
                for (i, (xi, yi)) in enumerate(zip(x, y)):
                    ax.plot(xi, yi, marker=(',', '3')[i % 2], color=c, lw=0.2, ms=3)

            elif dset == 'DIFF2MSD':
                ax.plot(x, y, 'o', markeredgecolor=c, markerfacecolor='none', markersize=5)

            else:
                ax.errorbar(x, y, yerr=yerr[0], color=c, label=label)

        if args.power:
            for j in range(i, len(power_law), len(args.input)):
                # plot power law
                pexp, pcoeff, px0, px1 = power_law[j]
                px = logspace(log10(px0), log10(px1), num=20)
                py = pcoeff * pow(px, pexp)
                ax.plot(px, py, '--', color=c)

    # optionally plot with logarithmic scale(s)
    if args.axes == 'xlog':
        ax.set_xscale('log')
    if args.axes == 'ylog':
        ax.set_yscale('log')
    if args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    l = ax.legend(loc=args.legend)
    l.legendPatch.set_alpha(0.7)

    if not title is None:
        plot.title(title)

    axlim = plot.axis('tight')
    if args.xlim:
        plot.xlim(args.xlim)
    if args.ylim:
        plot.ylim(args.ylim)

    plot.xlabel(r'$\tau$')
    ylabel = {
        'MSD': r'$\langle(r(t+\tau)-r(t))^2\rangle$',
        'MQD': r'$\langle(r(t+\tau)-r(t))^4\rangle$',
        'DIFF2MSD': r'$\frac{1}{2}\frac{d^2}{dt^2}\langle(r(t+\tau)-r(t))^2\rangle$',
        'VAC': r'$\langle v(t+\tau)v(t)\rangle$',
    }
    plot.ylabel(ylabel[dset])

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('corr', help='correlation functions')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlations file')
    parser.add_argument('--type', nargs='+', choices=['MSD', 'MQD', 'DIFF2MSD', 'VAC'], help='correlation function')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--unordered', action='store_true', help='disable block time ordering')
    parser.add_argument('--normalize', action='store_true', help='normalize function')
    parser.add_argument('--power', type=float, nargs='+', help='plot power law(s)')

