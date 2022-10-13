# -*- coding: utf-8 -*-
#
# pdf - compute and plot pair distribution function
#
# Copyright © 2008-2012  Felix Höfling and Peter Colberg
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
Compute and plot pair distribution function g(r)
"""
def plot(args):
    import os, os.path
    import h5py
    from matplotlib import pyplot as plt
    import h5mdtools._plot.label
    from numpy import linspace

    ax = plt.axes()
    label = None
    ax.axhline(y=1, color='black', lw=0.5)
    ax.set_color_cycle(args.colors)

    for (i, fn) in enumerate(args.input):
        try:
            f = h5py.File(fn, 'r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        try:

            # determine file type, prefer precomputed static structure factor data
            if 'structure' in list(f.keys()) and 'ssf' in list(f['structure'].keys()):
                import filon
                import h5mdtools._plot.ssf as ssf
                from scipy.constants import pi

                param = f['parameters']

                # load static structure factor from file
                H5 = f['structure/ssf/' + '/'.join(args.flavour)]
                q = f['structure/ssf/wavenumber'].__array__() # convert to NumPy array
                S_q, S_q_err = ssf.load_ssf(H5, args)

                # read some parameters
                dim = param['box'].attrs['dimension']
                density = param['box'].attrs['density']
                length = param['box'].attrs['length']

                # compute pair distribution function
                xlim = args.xlim or (0, min(length) / 2)
                r = linspace(xlim[0], xlim[1], num=args.bins)
                if r[0] == 0:
                    r = r[1:]
                if dim == 3:
                    # convert 3-dim Fourier transform F[S_q - 1] / (2π)³ to 1-dim Fourier integral
                    pdf = filon.filon(q * (S_q - 1), q, r).imag / (2 * pi * pi * r)
                    pdf_err = filon.filon(q * S_q_err, q, r).imag / (2 * pi * pi * r)
                pdf = 1 + pdf / density # add δ-contribution
                pdf_err = pdf_err / density

            elif 'particles' in list(f.keys()):
                # compute SSF from trajectory data
                H5 = f['particles/' + args.flavour[0]]
                r, pdf, pdf_err = pdf_from_trajectory(H5, args)
            else:
                raise SystemExit('Input file provides neither data for the static structure factor nor a trajectory')

            # before closing the file, store attributes for later use
            if 'param' in locals():
                attrs = h5mdtools._plot.label.attributes(param)

        except IndexError:
            raise SystemExit('invalid phase space sample offset')
        except KeyError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)
        finally:
            f.close()

        if args.label:
            label = args.label[i % len(args.label)] % attrs

        elif args.legend or not args.small:
            basename = os.path.splitext(os.path.basename(fn))[0]
            label = r'%s' % basename.replace('_', r'\_')

        if args.title:
            title = args.title % attrs

        c = args.colors[i % len(args.colors)]
        ax.plot(r, pdf, '-', color=c, label=label)
        if 'pdf_err' in locals():
            ax.errorbar(r, pdf, pdf_err, fmt='o', color=c, markerfacecolor=c, markeredgecolor=c, markersize=2, linewidth=.5)
        else:
            ax.plot(r, pdf, 'o', markerfacecolor=c, markeredgecolor=c, markersize=2)

        # write plot data to file
        if args.dump:
            f = open(args.dump, 'a')
            print('# %s, sample %s' % (label.replace(r'\_', '_'), args.sample), file=f)
            if 'pdf_err' in locals():
                print('# r   g(r)   g_err(r)', file=f)
                savetxt(f, array((r, pdf, pdf_err)).T)
            else:
                print('# r   g(r)', file=f)
                savetxt(f, array((r, pdf)).T)
            print('\n', file=f)
            f.close()

    # adjust axis ranges
    ax.axis('tight')
    if args.xlim:
        plt.setp(ax, xlim=args.xlim)
    if args.ylim:
        plt.setp(ax, ylim=args.ylim)
    else:
        plt.setp(ax, ylim=(0, plt.ylim()[1]))

    # optionally plot with logarithmic scale(s)
    if args.axes == 'xlog':
        ax.set_xscale('log')
    if args.axes == 'ylog':
        ax.set_yscale('log')
    if args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    plt.xlabel(args.xlabel or r'distance $r / \sigma$')
    plt.ylabel(args.ylabel or r'pair distribution function $g(r)$')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)

"""
Compute pair distribution function from trajectory data
"""
def pdf_from_trajectory(H5group, args):
    from scipy.constants import pi
    from scipy.special import gamma
    from numpy import array, diagonal, float32, histogram, power, prod, round_, sqrt, sum, zeros
    import re

    # read periodically extended particle positions,
    # read one or several samples, convert to single precision
    idx = [int(x) for x in re.split(':', args.sample)]
    data = H5group['position/value']
    if len(idx) == 1:
        samples = array([data[idx[0]],], dtype=float32)
    elif len(idx) == 2:
        samples = array(data[idx[0]:idx[1]], dtype=float32)
    elif len(idx) == 3:
        samples = array(data[idx[0]:idx[1]:idx[2]], dtype=float32)

    # positional coordinates dimension
    dim = H5group['box'].attrs['dimension']
    # periodic simulation box length
    length = diagonal(H5group['box/edges'])
    # number of particles
    N = data.shape[1]
    density = N / prod(length)

    r_max = args.xlim or (0, min(length) / 2)
    H = zeros(args.bins)
    for r in samples:
        for (i, j) in enumerate(range(r.shape[0] - 1, 0, -1)):
            # particle distance vectors
            dr = r[:j] - r[i + 1:]
            # minimum image distances
            dr = dr - round_(dr / length) * length
            # magnitude of distance vectors
            r_norm = sqrt(sum(dr * dr, axis=1))
            # accumulate histogram of minimum image distances
            h, bins = histogram(r_norm, bins=args.bins, range=r_max)
            H += 2 * h

    # volume of n-dimensional unit sphere
    Vn = power(pi, dim / 2.) / gamma(dim / 2. + 1.)
    # average number of atoms in ideal gas per interval
    n = Vn * density * (power(bins[1:], dim) - power(bins[:-1], dim))
    # compute pair distribution function g(r)
    pdf = H / samples.shape[0] / n / N
    pdf_err = sqrt(H) / samples.shape[0] / n / N

    return .5 * (bins[1:] + bins[:-1]), pdf, pdf_err

def add_parser(subparsers):
    parser = subparsers.add_parser('pdf', help='pair distribution function')
    parser.add_argument('input', nargs='+', metavar='INPUT', help='HDF5 file with trajectory or ssf data')
    parser.add_argument('--flavour', nargs=2, help='particle flavours')
    parser.add_argument('--sample', help='index of phase space sample(s)')
    parser.add_argument('--bins', type=int, help='number of histogram bins')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(flavour=('A', 'A'), sample='0', bins=50,)

