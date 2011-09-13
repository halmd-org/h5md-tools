# -*- coding: utf-8 -*-
#
# msv - macroscopic state variables
#
# Copyright © 2008-2011  Peter Colberg, Felix Höfling
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


"""
Plot macroscopic state variables
"""
def plot(args):
    import os, os.path
    from matplotlib import ticker
    from scipy import *
    from scipy.interpolate import interpolate
    import sys
    import h5py
    import h5md._plot.label

    from matplotlib import pyplot as plot

    dset_abbrev = {
        'ETOT': 'total_energy',
        'EPOT': 'potential_energy',
        'EKIN': 'kinetic energy',
        'ENHC': 'total_energy_nose_hoover_chain',
        'PRESS': 'pressure',
        'TEMP': 'temperature',
        'VCM': 'center_of_mass_velocity',
        'VZ': 'center_of_mass_velocity',
        'XVIR': 'hypervirial',
    }
    # equilibrium or stationary property
    if not args.type and not args.dataset:
        raise SystemExit('Either of the options --type or --dataset is required.')
    dset = args.type and dset_abbrev[args.type] or args.dataset

    ax = plot.axes()
    label = None
    title = None
    inset = None

    ylabel = predefined_label(args.type or dset)

    if args.inset:
        inset = plot.axes(args.inset)

    if args.zero:
        # plot zero line
        ax.axhline(y=0, color='black', lw=0.5)

    for i, fn in enumerate(args.input):
        try:
            f = h5py.File(fn, 'r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        try:
            H5 = f['observables']
            H5param = f['halmd']

            # positional coordinates dimension
            dim = H5param['box'].attrs['dimension']
            timestep = H5param.attrs['timestep']

            # open HDF5 datasets and convert to NumPy array
            #
            # This can be improved with respect to performance for the
            # case that only a small subset of the data is requested via args.xlim.
            x = asarray(H5[dset]['time'])
            y = asarray(H5[dset]['sample'])
            if args.xlim:
                idx = where((x >= args.xlim[0]) & (x <= args.xlim[1]))
                x, y = x[idx], y[idx]

            if args.type == 'VCM':
                # calculate center of velocity magnitude
                y = sqrt(sum(pow(y, 2), axis=-1))
            elif args.type == 'VZ':
                # select last velocity component
                y = y[:, dim - 1]
            elif args.type == 'ENHC':
                # add total energy to energy of chain variables
                y_ = H5['total_energy/sample']
                if y_.shape != H5[dset]['sample'].shape:
                    raise SystemExit('Sampling rates of total_energy and {0} disagree'.format(dset))
                if 'idx' in locals():
                    y_ = asarray(y_)[idx]
                y = y + asarray(y_)

            y_zero = y[0]

            y_mean, y_std = mean(y), std(y)

            if args.interpolate:
                fi = interpolate.interp1d(x, y)
                x = linspace(min(x), max(x), num=args.interpolate)
                y = fi(x)

            if args.label:
                attrs = h5md._plot.label.attributes(H5param)
                attrs['y_zero'] = r'%.2f' % y_zero
                attrs['y_mean'] = r'%.3f' % y_mean
                attrs['y_std'] = r'%#.2g' % y_std
                label = args.label[i % len(args.label)] % attrs
            elif args.legend or not args.small:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')
            if args.title:
                title = args.title % h5md._plot.label.attributes(H5param)

        except KeyError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)

        finally:
            f.close()

        if args.zero:
            # subtract zero value from data
            y = y - y[0];

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
            # plot values
            ax.text(x.min(), y_mean + 1.5*y_std, r'\parbox{1.2cm}{%s} = %.3g $\pm$ %.3g' 
                        % (ylabel[1], y_mean, y_std),
                    verticalalignment='bottom', horizontalalignment='left')
#            ax.text(0.75, 0.125, r'\parbox{1.2cm}{%s} = %.7g $\pm$ %.3g' 
#                        % (ylabel[1], y_mean, y_std),
#                    transform = ax.transAxes, verticalalignment='center',
#                    horizontalalignment='left')
            print '%.3g ± %.3g' % (y_mean, y_std)

    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_powerlimits((-1, 2))
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
    if args.zero:
        plot.setp(ax, ylabel=args.ylabel or ylabel[2])
    else:
        plot.setp(ax, ylabel=args.ylabel or ylabel[0])

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)

def predefined_label(name):
    label = {
        # name: absolute, mean, standard deviation, unit
        'ETOT': [
            r'$\langle E(t^*)\rangle / \epsilon$',
            r'$\langle\langle E\rangle\rangle_{t^*}$',
            r'$(\langle E(t^*)\rangle - \langle E(0)\rangle) / \epsilon$',
        ],
        'EPOT': [
            r'$\langle U(t^*)\rangle / \epsilon$',
            r'$\langle\langle U\rangle\rangle_{t^*}$',
            r'$(\langle U(t^*)\rangle - \langle U(0)\rangle) / \epsilon$',
        ],
        'EKIN': [
            r'$\langle T(t^*)\rangle / \epsilon$',
            r'$\langle\langle T\rangle\rangle_{t^*}$',
            r'$(\langle T(t^*)\rangle - \langle T(0)\rangle) / \epsilon$',
        ],
        'ENHC': [
            r'$\langle E_\text{NHC}(t^*)\rangle / \epsilon$',
            r'$\langle\langle E_\text{NHC}\rangle\rangle_{t^*}$',
            r'$(\langle E_\text{NHC}(t^*)\rangle - \langle E_\text{NHC}(0)\rangle) / \epsilon$',
        ],
        'PRESS': [
            r'$P^*(t^*)$',
            r'$\langle\langle P^*\rangle\rangle_{t^*}$',
            r'$P^*(t^*) - P^*(0)$',
        ],
        'TEMP': [
            r'$T^*(t^*)$',
            r'$\langle\langle T^*\rangle\rangle_{t^*}$',
            r'$T^*(t^*) - T^*(0)$',
        ],
        'VCM': [
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert$',
            r'$\langle\vert\langle \textbf{v}^*\rangle\vert\rangle_{t^*}$',
            r'$\vert\langle \textbf{v}^*(t^*)\rangle\vert - \vert\langle \textbf{v}^*(0)\rangle\vert$',
        ],
        'VZ': [
            r'$\langle v_z^*(t^*)\rangle$',
            r'$\langle\langle v_z^*\rangle\rangle_{t^*}$',
            r'$\langle v_z^*(t^*)\rangle - \langle v_z^*(0)\rangle$',
        ],
        'XVIR': [
            r'$\langle X^*(t^*)\rangle$',
            r'$\langle\langle X^*\rangle\rangle_{t^*}$',
            r'$\langle X^*(t^*)\rangle - \langle X^*(0)\rangle$',
        ],
    }

    if name in label.keys():
        return label[name]
    else:
        return [
            name,
            r'$\langle \text{' + name + '} \rangle$',
            r'$\Delta \text{' + name + '}$',
        ]


def add_parser(subparsers):
    parser = subparsers.add_parser('msv', help='macroscopic state variables')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='H5MD input file')
    parser.add_argument('--dataset', help='specify dataset')
    parser.add_argument('--type', choices=['ETOT', 'EPOT', 'EKIN', 'ENHC', 'PRESS', 'TEMP', 'VCM', 'VZ', 'XVIR'], help='equilibrium or stationary property')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--mean', action='store_true', help='plot mean and standard deviation')
    parser.add_argument('--zero', action='store_true', help='substract zero value')
    parser.add_argument('--interpolate', type=int, help='linear interpolation to given number of plot points')
    parser.add_argument('--inset', metavar='VALUE', type=float, nargs=4, help='plot inset')
    parser.add_argument('--inset-xlim', metavar='VALUE', type=float, nargs=2, help='limit inset x-axis to given range')
    parser.add_argument('--inset-ylim', metavar='VALUE', type=float, nargs=2, help='limit inset y-axis to given range')
    parser.add_argument('--inset-xlabel', help='inset x-axis label')
    parser.add_argument('--inset-ylabel', help='inset y-axis label')

