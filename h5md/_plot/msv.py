# -*- coding: utf-8 -*-
#
# msv - macroscopic state variables
#
# Copyright © 2008-2014 Felix Höfling
# Copyright © 2008-2011 Peter Colberg
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

# dictionary with dataset abbreviations
dset_abbrev = {
    'PRESS': 'pressure',
    'TEMP': 'temperature',
    'DENS': 'density',
    'EPOT': 'potential_energy',
    'EKIN': 'kinetic_energy',
    'EINT': 'internal_energy',
    'ENHC': 'nose_hoover_chain/internal_energy',
    'ENTH': 'enthalpy',
    'VCM': 'center_of_mass_velocity',
    'VX': 'center_of_mass_velocity',
    'VY': 'center_of_mass_velocity',
    'VZ': 'center_of_mass_velocity',
}

def plot(args):
    """
    Plot macroscopic state variables
    """
    import os, os.path
    from matplotlib import ticker
    from numpy import asarray, floor, in1d, intersect1d, linspace, mean, std, where
    from scipy.interpolate import interpolate
    import sys
    import h5py
    import h5md._plot.label

    from matplotlib import pyplot as plot

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
        # open H5MD file, version ≥ 1.0
        try:
            f = h5py.File(fn, 'r')
            version = f['h5md'].attrs['version']
            assert(version[0] == 1 and version[1] >= 0)
        except (AssertionError, IOError, KeyError):
            raise SystemExit("failed to open H5MD (≥ 1.0) file: {0:s}".format(fn))

        # check for the thermodynamics module ≥ 1.0
        try:
            version = f['h5md/modules/thermodynamics'].attrs['version']
            assert(version[0] == 1 and version[1] >= 0)
        except (AssertionError, KeyError):
            raise SystemExit("thermodynamics module (≥ 1.0) not present in H5MD file: {0:s}".format(fn))

        if not 'observables' in f.keys():
            raise SystemExit("missing /observables group in file: {0:s}".format(fn))
        H5 = f['observables']

        if args.group:
            try:
                H5 = H5[args.group]
            except KeyError:
                raise SystemExit("missing group /observables/{0:s} in file: {1:s}".format(args.group, fn))

        if not dset in H5.keys():
            raise SystemExit("missing H5MD element {0:s}/{1:s} in file: {2:s}".format(H5.name, dset, fn))

        H5element = H5[dset]
        if type(H5element) != h5py.Group:
            raise SystemExit("H5MD element {0:s} is time-independent".format(H5element.name))

        try:
            # open HDF5 datasets and convert to NumPy array
            #
            # This can be improved with respect to performance for the
            # case that only a small subset of the data is requested via args.xlim.
            x = asarray(H5element['time'])
            y = asarray(H5element['value'])
            step = asarray(H5element['step'])
            if args.xlim:
                idx = where((x >= args.xlim[0]) & (x <= args.xlim[1]))
                x, y, step = x[idx], y[idx], step[idx]

            if args.type == 'VCM':
                # calculate center of velocity magnitude
                y = sqrt(sum(pow(y, 2), axis=-1))
            elif args.type == 'VX':
                # select first velocity component
                y = y[:, 0]
            elif args.type == 'VY':
                # select second velocity component
                y = y[:, 1]
            elif args.type == 'VZ':
                # select third velocity component
                y = y[:, 2]
            elif args.type == 'ENHC':
                # add internal energy to energy of chain variables
                # deal with possibly different sampling intervals of the two data sets
                y_ = H5['internal_energy/value']
                step_ = H5['internal_energy/step']

                # form intersection of both 'step' sets and construct indexing arrays
                step_intersect = intersect1d(step, step_)
                idx = where(in1d(step, step_intersect))
                idx_ = where(in1d(step_, step_intersect))

                # restrict x, y to points that appear in y_ as well
                x, y = x[idx], y[idx]
                y = y + asarray(y_)[idx_]   # finally add data

            y_zero = y[0]

            y_mean, y_std = mean(y), std(y)

            if args.label:
                if 'parameters' in f.keys:
                    attrs = h5md._plot.label.attributes(f['parameters'])
                attrs['y_zero'] = r'%.2f' % y_zero
                attrs['y_mean'] = r'%.3f' % y_mean
                attrs['y_std'] = r'%#.2g' % y_std
                label = args.label[i % len(args.label)] % attrs
            elif args.legend or not args.small:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')
            if args.title and 'parameters' in f.keys:
                title = args.title % h5md._plot.label.attributes(f['parameters'])

        except KeyError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)

        finally:
            f.close()

        # subtract zero value from data
        if args.zero:
            y = y - y[0];

        # divide data in blocks of given size and compute block averages
        if args.block_average:
            block_size = args.block_average
            shape = list(x.shape)
            if shape[0] > block_size:
                # cut arrays to extent of first dimension being a multiple of nblocks
                a = int(block_size * floor(shape[0] / block_size))
                x, y = x[:a], y[:a]
                shape[0] = block_size
            shape.insert(0, -1)
            x = mean(reshape(x, shape), axis=1)
            y = mean(reshape(y, shape), axis=1)

        # put data on linear grid using linear interpolation
        if args.points:
            fi = interpolate.interp1d(x, y)
            x = linspace(min(x), max(x), num=args.points)
            y = fi(x)

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
        'EINT': [
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
        'VX': [
            r'$\langle v_x^*(t^*)\rangle$',
            r'$\langle\langle v_x^*\rangle\rangle_{t^*}$',
            r'$\langle v_x^*(t^*)\rangle - \langle v_x^*(0)\rangle$',
        ],
        'VY': [
            r'$\langle v_y^*(t^*)\rangle$',
            r'$\langle\langle v_y^*\rangle\rangle_{t^*}$',
            r'$\langle v_y^*(t^*)\rangle - \langle v_y^*(0)\rangle$',
        ],
        'VZ': [
            r'$\langle v_z^*(t^*)\rangle$',
            r'$\langle\langle v_z^*\rangle\rangle_{t^*}$',
            r'$\langle v_z^*(t^*)\rangle - \langle v_z^*(0)\rangle$',
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
    parser.add_argument('--group', help='specify particle group')
    parser.add_argument('--type', choices=dset_abbrev.keys(), help='equilibrium or stationary property')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--mean', action='store_true', help='plot mean and standard deviation')
    parser.add_argument('--zero', action='store_true', help='substract zero value')
    parser.add_argument('--points', type=int, help='number of plot points (linear interpolation)')
    parser.add_argument('--block-average', type=int, help='plot block averages of given block size')
    parser.add_argument('--inset', metavar='VALUE', type=float, nargs=4, help='plot inset')
    parser.add_argument('--inset-xlim', metavar='VALUE', type=float, nargs=2, help='limit inset x-axis to given range')
    parser.add_argument('--inset-ylim', metavar='VALUE', type=float, nargs=2, help='limit inset y-axis to given range')
    parser.add_argument('--inset-xlabel', help='inset x-axis label')
    parser.add_argument('--inset-ylabel', help='inset y-axis label')

