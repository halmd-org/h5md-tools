# -*- coding: utf-8 -*-
#
# tcf - time correlation functions
#
# Copyright © 2013 Felix Höfling
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

def plot(args):
    """
    Plot time correlation functions
    """
    import h5py
    import h5md._plot.label
    import numpy as np

    from matplotlib import pyplot as plot
    from matplotlib import ticker

    ax = plot.axes()

    # translate abbreviations for dataset name
    dset_abbrev = {
        'MSD': 'mean_square_displacement',
        'MQD': 'mean_quartic_displacement',
        'VACF': 'velocity_autocorrelation',
        'ISF': 'intermediate_scattering_function',
    }
    if not args.type and not args.dataset:
        raise SystemExit('Either of the options --type or --dataset is required.')
    dset = args.type and dset_abbrev[args.type] or args.dataset

    for i, fn in enumerate(args.input):
        with h5py.File(fn, 'r') as f:
            h5 = f['dynamics']
            if args.group:
                h5 = h5[args.group]
            h5 = h5[dset]

            # open HDF5 datasets,
            # convert time dataset to NumPy array
            x = np.asarray(h5['time'])
            y = 'value' in h5 and h5['value'] or h5['mean'] # FIXME
            yerr = 'error' in h5 and h5['error'] or None
            parameter = 'wavenumber' in h5 and h5['wavenumber'] or None
#            x, y, yerr = x[0], y[0], yerr[0]

            # apply parameter slice
            rank = len(y.shape) - len(x.shape) # tensor rank of TCF
            if args.slice:
                if rank != 1:
                    raise SystemExit("Correlation function not eligible for parameter slicing")
                s = slice(*args.slice)
                y = y[..., s]
                if yerr != None:
                    yerr = yerr[..., s]
                if parameter != None:
                    parameter = parameter[s]

            # convert to NumPy arrays before closing the HDF5 file
            y = np.asarray(y)
            if yerr != None:
                yerr = np.asarray(yerr)
            if parameter != None:
                parameter = np.asarray(parameter)

        # blockwise normalisation
        if args.norm:
            # normalise with data at t=0,
            # distinguish between flat and blocked time data
            norm = y[np.where(x == 0)]
            if len(x.shape) == 1:
                norm = norm.reshape((1,) + y.shape[1:])
            elif len(x.shape) == 2:
                norm = norm.reshape((y.shape[0],) + (1,) + y.shape[2:])
            y = y / norm
            if yerr != None:
                yerr = yerr / norm
            assert((y[np.where(x==0)] == 1).all)

        # flatten time coordinate due to block structure
        y = y.reshape((-1,) + y.shape[len(x.shape):])
        if yerr != None:
            yerr = yerr.reshape((-1,) + yerr.shape[len(x.shape):])
        x = x.flatten()

        # sort data by ascending time
        idx = x.argsort(kind='mergesort')
        x, y = x[idx], y[idx]
        if yerr != None:
            yerr = yerr[idx]

        if parameter == None or len(parameter) == 1:
            c = args.colors[i % len(args.colors)] # cycle plot color
            ax.plot(x, y, color=c, label=fn)
            if yerr != None:
                ax.errorbar(x, y, yerr=yerr, color=c, mec=c, mfc=c)
        else:
            for j,p in enumerate(parameter):
                c = args.colors[j % len(args.colors)] # cycle plot color
                label = (i == 0) and '{0:3g}'.format(p) or None
                ax.plot(x, y[:, j], color=c, label=label)
                if yerr != None:
                    ax.errorbar(x, y[:, j], yerr=yerr[:, j], color=c, mec=c, mfc=c)

    if args.legend or not args.small:
        ax.legend(loc=args.legend)

    # set plot limits
    ax.axis('tight')
    if args.xlim:
        plot.setp(ax, xlim=args.xlim)
    if args.ylim:
        plot.setp(ax, ylim=args.ylim)

    # optionally plot with logarithmic scale(s)
    if args.axes == 'xlog':
        ax.set_xscale('log')
    if args.axes == 'ylog':
        ax.set_yscale('log')
    if args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    plot.setp(ax, xlabel=args.xlabel or 'time $t$')
    if args.norm:
        plot.setp(ax, ylabel=args.ylabel or r'$C(t) / C(0)$')
    else:
        plot.setp(ax, ylabel=args.ylabel or r'$C(t)$')

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('tcf', help='time correlation functions')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='H5MD input file')
    parser.add_argument('--dataset', help='specify dataset')
    parser.add_argument('--group', help='specify particle group')
    parser.add_argument('--type', choices=['MSD', 'MQD', 'VACF', 'ISF'], help='time correlation function')
    parser.add_argument('--slice', nargs='+', type=int, help='slicing index for the parameter, e.g., wavenumber')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', default='xlog', choices=['linear', 'xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--norm', action='store_true', help='normalise correlation function by t=0 value')
