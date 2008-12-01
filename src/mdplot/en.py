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
    from matplotlib import pyplot as plot

    # thermal equilibrium property
    dset = args.type

    ylabel = {
        # type: absolute, mean, standard deviation, unit
        'ETOT': [
            r'$\langle E(t^*)\rangle / \epsilon$',
            r'$\langle\langle E\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle E\rangle}$',
            r'$\dfrac{\langle E(t^*)\rangle - \langle E(0)\rangle}{\delta t^2\epsilon}$',
            r'$(\langle E(t^*)\rangle - \langle E(0)\rangle) / \epsilon$',
        ],
        'EPOT': [
            r'$\langle U(t^*)\rangle / \epsilon$',
            r'$\langle\langle U\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle U\rangle}$',
            r'$\dfrac{\langle U(t^*)\rangle - \langle U(0)\rangle}{\delta t^2 \epsilon}$',
            r'$(\langle U(t^*)\rangle - \langle U(0)\rangle) / \epsilon$',
        ],
        'EKIN': [
            r'$\langle T(t^*)\rangle / \epsilon$',
            r'$\langle\langle T\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle T\rangle}$',
            r'$\dfrac{\langle T(t^*)\rangle - \langle T(0)\rangle}{\delta t^2\epsilon}$',
            r'$(\langle T(t^*)\rangle - \langle T(0)\rangle) / \epsilon$',
        ],
        'PRESS': [
            r'$\langle P^*(t^*)\rangle$',
            r'$\langle\langle P^*\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle P^*\rangle}$',
            r'$\dfrac{\langle P^*(t^*)\rangle - \langle P^*(0)\rangle}{\delta t^2}$',
            r'$\langle P^*(t^*)\rangle - \langle P^*(0)\rangle$',
        ],
        'TEMP': [
            r'$\langle T^*(t^*)\rangle$',
            r'$\langle\langle T^*\rangle\rangle_{t^*}$',
            r'$\sigma_{\langle T^*\rangle}$',
            r'$\dfrac{\langle T^*(t^*)\rangle - \langle T^*(0)\rangle}{\delta t^2}$',
            r'$\langle T^*(t^*)\rangle - \langle T^*(0)\rangle$',
        ],
        'VCM': [
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert$',
            r'$\langle\vert\langle \textbf{v}^*\rangle\vert\rangle_{t^*}$',
            r'$\sigma_{\vert\langle \textbf{v}^*\rangle\vert}$',
            r'$\dfrac{\vert\langle \textbf{v}^*(t^*)\rangle\vert - \vert\langle \textbf{v}^*(0)\rangle\vert}{\delta t^2}$',
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert - \vert\langle \textbf{v}^*(0)\rangle\vert$',
        ],
    }

    ax = plot.axes()
    label = None
    title = None
    inset = None

    if args.inset:
        inset = plot.axes(args.inset)

    if args.zero or args.rescale:
        # plot zero line
        ax.axhline(y=0, color='black', lw=0.5)

    for i, fn in enumerate(args.input):
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        try:
            data = H5._v_children[dset]
            x = data[:, 0]
            if dset == 'VCM':
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
            version = H5.param.program._v_attrs.version

            # work around sampling bug yielding zero potential energy at time zero
            if dset in ('ETOT', 'EPOT', 'PRESS') and version < 'v0.2.1-21-g16e09fc':
                print >> sys.stderr, 'WARNING: detected buggy ljfluid version, discarding sample at time zero'
                x, y = x[1:], y[1:]

            if args.xlim:
                xi = where((x >= args.xlim[0]) & (x <= args.xlim[1]))
                x, y = x[xi], y[xi]

            y_mean, y_std = mean(y), std(y)

            if args.interpolate:
                fi = interpolate.interp1d(x, y)
                x = linspace(min(x), max(x), num=args.interpolate)
                y = fi(x)

            if args.label:
                attrs = mdplot.label.attributes(H5.param)
                attrs['y_mean'] = r'%.3f' % y_mean
                attrs['y_std'] = r'%#.2g' % y_std
                label = args.label[i % len(args.label)] % attrs
            elif args.legend or not args.small:
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

        if args.inset:
            inset.plot(x, y, color=c)

        if args.mean:
            # plot standard deviation
            ax.axhspan(y_mean - y_std, y_mean + y_std, facecolor=c, edgecolor=c, alpha=0.1)
            # plot mean
            ax.axhline(y_mean, linestyle='--', color=c, alpha=0.5)
            ax.text(1.01 * x.max() - 0.01 * x.min(), y_mean, r'%y_std' % ylabel[dset][1],
                    verticalalignment='center', horizontalalignment='left')
            # plot values
            ax.text(0.75, 0.125, r'\parbox{1.2cm}{%y_std} = %.3g' % (ylabel[dset][1], y_mean),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')
            ax.text(0.75, 0.075, r'\parbox{1.2cm}{%y_std} = %.3g' % (ylabel[dset][2], y_std),
                    transform = ax.transAxes, verticalalignment='center',
                    horizontalalignment='left')

    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(major_formatter)
    if args.inset:
        inset.yaxis.set_major_formatter(major_formatter)

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

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

    if not title is None:
        plot.title(title)

    plot.setp(ax, xlabel=args.xlabel or r'$t^*$')
    if args.rescale:
        plot.setp(ax, ylabel=args.ylabel or ylabel[dset][3])
    elif args.zero:
        plot.setp(ax, ylabel=args.ylabel or ylabel[dset][4])
    else:
        plot.setp(ax, ylabel=args.ylabel or ylabel[dset][0])

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


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
    parser.add_argument('--inset', metavar='VALUE', type=float, nargs=4, help='plot inset')
    parser.add_argument('--inset-xlim', metavar='VALUE', type=float, nargs=2, help='limit inset x-axis to given range')
    parser.add_argument('--inset-ylim', metavar='VALUE', type=float, nargs=2, help='limit inset y-axis to given range')
    parser.add_argument('--inset-xlabel', help='inset x-axis label')
    parser.add_argument('--inset-ylabel', help='inset y-axis label')

