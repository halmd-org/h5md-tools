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
    label = None
    title = None
    inset = None

    if not args.axes in ('ylog', 'loglog'):
        # plot zero line
        ax.axhline(y=0, color='black', lw=0.5)

    if args.power_inset:
        if 'MSD' in args.type:
            inset = plot.axes([0.55, 0.18, 0.33, 0.25])
        else:
            inset = plot.axes([0.66, 0.66, 0.22, 0.22])

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

                if dset in ('DIFF2MSD', 'DIFFMSD'):
                    if dset == 'DIFF2MSD':
                        # calculate VACF from 2nd discrete derivative of MSD
                        h = data[:, 2:, 0] - data[:, :-2, 0]
                        x = (data[:, 2:, 0] + data[:, :-2, 0]) / 2
                        y = 0.5 * diff(data[:, :, 1], axis=1, n=2) / pow(h, 2)
                        x, x0 = reshape(x[:, 1:], (-1, )), x[0, 0]
                        y, y0 = reshape(y[:, 1:], (-1, )), y[0, 0]
                    else:
                        # calculate diffusion constant from central difference of MSD
                        h = data[:, 1:, 0] - data[:, :-1, 0]
                        x = (data[:, 1:, 0] + data[:, :-1, 0]) / 2
                        y = diff(data[:, :, 1], axis=1, n=1) / (6 * h)
                        x, x0 = reshape(x[:, 1:], (-1, )), x[0, 0]
                        y, y0 = reshape(y[:, 1:], (-1, )), y[0, 0]

                    if not args.unordered:
                        # prepend time zero from lowest block
                        x = append(x0, x)
                        y = append(y0, y)
                        # time-order correlation function samples
                        time_order = x.argsort()
                        x, y = x[time_order], y[time_order]

                    if dset == 'DIFF2MSD' and args.normalize:
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
                elif args.legend or not args.small:
                    basen = os.path.splitext(os.path.basename(fn))[0]
                    label = r'%s:%s' % (dset, basen.replace('_', r'\_'))
                if args.title:
                    title = args.title % mdplot.label.attributes(H5.param)

            except tables.exceptions.NoSuchNodeError:
                raise SystemExit('missing simulation data in file: %s' % fn)

            finally:
                f.close()

            _y = y
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
                ax.plot(x[_y > 0], y[_y > 0], '+', markeredgecolor=c, markerfacecolor='none', markersize=5)
                ax.plot(x[_y < 0], y[_y < 0], 'o', markeredgecolor=c, markerfacecolor='none', markersize=5)
                if args.power_inset:
                    py = y * pow(x, -args.power_inset)
                    inset.plot(x, py, 'o', markeredgecolor=c, markerfacecolor='none', markersize=3)

            elif dset == 'DIFFMSD':
                ax.plot(x, y, color=c, label=label)

            else:
                ax.errorbar(x, y, yerr=yerr[0], color=c, label=label)
                if args.power_inset:
                    py = y * pow(x, -args.power_inset)
                    inset.plot(x, py, color=c)

    # optionally plot power laws
    if args.power_law:
        p = reshape(args.power_law, (-1, 4))
        for (pow_exp, pow_coeff, pow_xmin, pow_xmax) in p:
            px = logspace(log10(pow_xmin), log10(pow_xmax), num=100)
            py = pow_coeff * pow(px, pow_exp)
            ax.plot(px, py, 'k--')

    # optionally plot with logarithmic scale(s)
    if args.axes == 'xlog':
        ax.set_xscale('log')
        if inset:
            inset.set_xscale('log')
    if args.axes == 'ylog':
        ax.set_yscale('log')
        if inset:
            inset.set_yscale('log')
    if args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
        if inset:
            inset.set_xscale('log')
            inset.set_yscale('log')

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

    if inset:
        inset.axis('tight')
        if args.inset_xlim:
            plot.setp(inset, xlim=args.inset_xlim)
        if args.inset_ylim:
            plot.setp(inset, ylim=args.inset_ylim)
        if args.inset_xlabel:
            plot.setp(inset, xlabel=args.inset_xlabel)
        if args.inset_ylabel:
            plot.setp(inset, ylabel=args.inset_ylabel)

    plot.setp(ax, xlabel=args.xlabel or r'$t^*$')
    ylabel = {
        'MSD': r'$\langle\delta r(t^*)^2\rangle\sigma^{-2}$',
        'MQD': r'$\langle\delta r(t^*)^4\rangle\sigma^{-4}$',
        'DIFFMSD': r'$\frac{1}{6}\frac{d}{dt}\langle\delta r(t^*)^2\rangle\sigma^{-2}$',
        'DIFF2MSD': r'$\frac{1}{2}\frac{d^2}{dt^2}\langle\delta r(t^*)^2\rangle$',
        'VAC': r'$\langle v(t^*)v(0)\rangle$',
    }
    plot.setp(ax, ylabel=args.ylabel or ylabel[dset])

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('corr', help='correlation functions')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlations file')
    parser.add_argument('--type', nargs='+', choices=['MSD', 'DIFFMSD', 'DIFF2MSD', 'MQD', 'VAC'], help='correlation function')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--unordered', action='store_true', help='disable block time ordering')
    parser.add_argument('--normalize', action='store_true', help='normalize function')
    parser.add_argument('--power-law', type=float, nargs='+', help='plot power law curve(s)')
    parser.add_argument('--power-inset', type=float, help='plot power law inset')
    parser.add_argument('--inset-xlim', metavar='VALUE', type=float, nargs=2, help='limit inset x-axis to given range')
    parser.add_argument('--inset-ylim', metavar='VALUE', type=float, nargs=2, help='limit inset y-axis to given range')
    parser.add_argument('--inset-xlabel', help='inset x-axis label')
    parser.add_argument('--inset-ylabel', help='inset y-axis label')

