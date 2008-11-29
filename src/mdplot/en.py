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
from scipy import *
import sys
import tables
import mdplot.label


"""
Plot mean total energy per particle
"""
def plot(args):
    from matplotlib import pyplot as plt

    # thermal equilibrium property
    tep = args.type

    ylabel = {
        # type: absolute, mean, standard deviation, unit
        'ETOT': [
            r'$\langle E(t^*)\rangle / \epsilon$',
            r'$\langle\langle E\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle E\rangle}$',
            r'$\dfrac{\langle E(t^*)\rangle - \langle E(0)\rangle}{\delta t^2\epsilon}$',
            r'$\dfrac{\langle E(t^*)\rangle - \langle E(0)\rangle}{\epsilon}$',
        ],
        'EPOT': [
            r'$\langle U(t^*)\rangle / \epsilon$',
            r'$\langle\langle U\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle U\rangle}$',
            r'$\dfrac{\langle U(t^*)\rangle - \langle U(0)\rangle}{\delta t^2 \epsilon}$',
            r'$\dfrac{\langle U(t^*)\rangle - \langle U(0)\rangle}{\epsilon}$',
        ],
        'EKIN': [
            r'$\langle T(t^*)\rangle / \epsilon$',
            r'$\langle\langle T\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle T\rangle}$',
            r'$\dfrac{\langle T(t^*)\rangle - \langle T(0)\rangle}{\delta t^2\epsilon}$',
            r'$\dfrac{\langle T(t^*)\rangle - \langle T(0)\rangle}{\epsilon}$',
        ],
        'PRESS': [
            r'$\langle P(t^*)\rangle$',
            r'$\langle\langle P\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle P\rangle}$',
            r'$\dfrac{\langle P(t^*)\rangle - \langle P(0)\rangle}{\delta t^2}$',
            r'$\langle P(t^*)\rangle - \langle P(0)\rangle$',
        ],
        'TEMP': [
            r'$\langle T(t^*)\rangle$',
            r'$\langle\langle T\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle T\rangle}$',
            r'$\dfrac{\langle T(t^*)\rangle - \langle T(0)\rangle}{\delta t^2}$',
            r'$\langle T(t^*)\rangle - \langle T(0)\rangle$',
        ],
        'VCM': [
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert$',
            r'$\langle\vert\langle \textbf{v}^*\rangle\vert\rangle_{t^*}$',
            r'$\sigma_{\vert\langle \textbf{v}^*\rangle\vert}$',
            r'$\dfrac{\vert\langle \textbf{v}^*(t^*)\rangle\vert - \vert\langle \textbf{v}^*(0)\rangle\vert}{\delta t^2}$',
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert - \vert\langle \textbf{v}^*(0)\rangle\vert$',
        ],
    }

    ax = plt.axes()
    title = None

    for i, fn in enumerate(args.input):
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        try:
            data = H5._v_children[tep]
            x = data[:, 0]
            if tep == 'VCM':
                # positional coordinates dimension
                dim = H5.param.mdsim._v_attrs.dimension
                # calculate center of velocity magnitude
                if dim == 3:
                    y = sqrt(data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2] + data[:, 3] * data[:, 3])
                else:
                    y = sqrt(data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2])
            else:
                y = data[:, 1]
            timestep = H5.param.mdsim._v_attrs.timestep

            if args.interpolate:
                fi = interpolate.interp1d(x, y)
                x = linspace(min(x), max(x), num=args.interpolate)
                y = fi(x)

            if args.label:
                label = args.label[i % len(args.label)] % mdplot.label.attributes(H5.param)
            else:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')
            if args.title:
                title = args.title % mdplot.label.attributes(H5.param)

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % fn)

        finally:
            f.close()

        if args.zero or args.rescale:
            # subtract zero value from data
            y = y - y[0];
        if args.rescale:
            # divide by squared timestep
            y = y / pow(timestep, 2)

        if not len(x) or not len(y):
            raise SystemExit('empty plot range')

        # cycle plot color
        c = args.colors[i % len(args.colors)]
        ax.plot(x, y, color=c, label=label)

        if args.mean:
            m, s = mean(y), std(y)
            # plot standard deviation
            ax.axhspan(m - s, m + s, facecolor=c, edgecolor=c, alpha=0.1)
            # plot mean
            ax.axhline(m, linestyle='--', color=c, alpha=0.5)
            ax.text(1.01 * x.max() - 0.01 * x.min(), m, r'%s' % ylabel[tep][1],
                    verticalalignment='center', horizontalalignment='left')
            # plot values
            ax.text(0.75, 0.125, r'\parbox{1.2cm}{%s} = %.3g' % (ylabel[tep][1], m),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')
            ax.text(0.75, 0.075, r'\parbox{1.2cm}{%s} = %.3g' % (ylabel[tep][2], s),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')

    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(major_formatter)

    l = ax.legend(loc=args.legend)
    l.legendPatch.set_alpha(0.7)

    plt.axis('tight')
    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    if not title is None:
        plt.title(title)
    plt.xlabel(args.xlabel or r'$t^*$')
    if args.rescale:
        plt.ylabel(args.ylabel or ylabel[tep][3])
    elif args.zero:
        plt.ylabel(args.ylabel or ylabel[tep][4])
    else:
        plt.ylabel(args.ylabel or ylabel[tep][0])

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('en', help='thermal equilibrium properties')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 energy file')
    parser.add_argument('--type', required=True, choices=['ETOT', 'EPOT', 'EKIN', 'PRESS', 'TEMP', 'VCM'], help='thermal equilibrium property')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--mean', action='store_true', help='plot mean and standard deviation')
    parser.add_argument('--zero', action='store_true', help='substract zero value')
    parser.add_argument('--rescale', action='store_true', help='substract zero value and divide by squared timestep')
    parser.add_argument('--interpolate', type=int, help='linear interpolation to given number of plot points')

