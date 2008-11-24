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
import mdplot.label


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
    slabel = {
        'ETOT':  r'$\sigma_{\langle E\rangle}$',
        'EPOT':  r'$\sigma_{\langle U\rangle}$',
        'EKIN':  r'$\sigma_{\langle T\rangle}$',
        'PRESS': r'$\sigma_{\langle P\rangle}$',
        'TEMP':  r'$\sigma_{\langle T\rangle}$',
        'VCM':   r'$\sigma_{\vert\langle \textbf{v}^*\rangle\vert}$',
    }

    ax = plt.axes()
    title = None

    ci = 0
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

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

            if args.label:
                label = args.label % mdplot.label.attributes(H5.param)
            else:
                label = fn.replace('_', r'\_')
            if args.title:
                title = args.title % mdplot.label.attributes(H5.param)

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % fn)

        finally:
            f.close()

        if args.xaxis:
            # limit data points to given x-axis range
            i = numpy.where((x >= args.xaxis[0]) & (x <= args.xaxis[1]))
            x, y = x[i], y[i]
        if args.yaxis:
            # limit data points to given y-axis range
            i = numpy.where((y >= args.yaxis[0]) & (y <= args.yaxis[1]))
            x, y = x[i], y[i]

        if not len(x) or not len(y):
            raise SystemExit('empty plot range')

        # cycle plot color
        c = args.colors[ci % len(args.colors)]
        ci += 1
        ax.plot(x, y, color=c, label=label)

        if args.mean:
            m, s = numpy.mean(y), numpy.std(y)
            # plot standard deviation
            ax.axhspan(m - s, m + s, facecolor=c, edgecolor=c, alpha=0.1)
            # plot mean
            ax.axhline(m, linestyle='--', color=c, alpha=0.5)
            ax.text(1.01 * x.max() - 0.01 * x.min(), m, mlabel[args.type],
                    verticalalignment='center', horizontalalignment='left')
            # plot values
            ax.text(0.75, 0.125, r'\parbox{1.2cm}{%s} = %.3g' % (mlabel[args.type], m),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')
            ax.text(0.75, 0.075, r'\parbox{1.2cm}{%s} = %.3g' % (slabel[args.type], s),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')

    major_formatter = ticker.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    l = ax.legend(loc=args.legend, labelsep=0.01, pad=0.1, axespad=0.025)
    l.legendPatch.set_alpha(0.7)

    if not title is None:
        plt.title(title)
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
    parser.add_argument('--yaxis', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--mean', action='store_true', help='plot mean and standard deviation')

