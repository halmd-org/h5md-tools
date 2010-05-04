# -*- coding: utf-8 -*-
#
# mdplot - Molecular Dynamics simulation plotter
#
# Copyright © 2008-2010  Peter Colberg, Felix Höfling
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
from numpy import *
from scipy.special import gamma
import mdplot.label
import sys
import tables


"""
Plot pair distribution function

M.P. Allen and D.J. Tildesley,
Computer Simulation of Liquids, 1989,
Oxford University Press, pp. 55, 183-184
"""
def plot(args):
    from matplotlib import pyplot as plot

    ax = plot.axes()
    label = None
    ax.axhline(y=1, color='black', lw=0.5)
    ax.set_color_cycle(args.colors)

    for k, fn in enumerate(args.input):
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        # HDF5 root group
        H5 = f.root

        try:
            try:
                # particle positions of phase space sample
                if args.flavour:
                    r = H5.trajectory._v_children[args.flavour].r[args.sample]
                else:
                    r = H5.trajectory.r[args.sample]
                # simulation time
                time = H5.trajectory.t[args.sample]
            except IndexError:
                raise SystemExit('out-of-bounds phase space sample number')

            # periodic simulation box length
            box = H5.param.mdsim._v_attrs.box_length
            # number of particles
            N = H5.param.mdsim._v_attrs.particles
            if not isscalar(N):
                N = N[ord(args.flavour[:1]) - ord('A')]

            # positional coordinates dimension
            dim = H5.param.mdsim._v_attrs.dimension
            # particle density
            density = H5.param.mdsim._v_attrs.density

            cutoff = args.xlim or (0, box / 2)
            H = zeros(args.bins)
            for (i, j) in enumerate(range(N - 1, 0, -1)):
                # particle distance vectors
                dr = r[:j] - r[i + 1:]
                # minimum image distances
                dr = dr - round_(dr / box) * box
                # squared minimum image distances
                rr = dr[:, 0] * dr[:, 0] + dr[:, 1] * dr[:, 1]
                if dim == 3:
                    rr = rr + dr[:, 2] * dr[:, 2]
                # accumulate histogram of minimum image distances
                h, bins = histogram(sqrt(rr), bins=args.bins, range=cutoff, new=True)
                H = H + 2 * h

            if args.label:
                label = args.label[k % len(args.label)] % mdplot.label.attributes(H5.param)
            elif args.legend or not args.small:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')

        except tables.exceptions.NoSuchNodeError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)

        finally:
            f.close()

        # volume of n-dimensional unit sphere
        Vn = power(pi, dim / 2.) / gamma(dim / 2. + 1.)
        # average number of atoms in ideal gas per interval
        n = Vn * density * (power(bins[1:], dim) - power(bins[:-1], dim))
        # compute pair correlation function
        g = H / n / N

        ax.plot(bins[:-1], g, marker='.', label=label)

    ax.axis('tight')
    if args.xlim:
        plot.setp(ax, xlim=args.xlim)
    if args.ylim:
        plot.setp(ax, ylim=args.ylim)

    plot.setp(ax, xlabel=args.xlabel or r'$r / \sigma$')
    plot.setp(ax, ylabel=args.ylabel or r'$g(r)$')
    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('pdf', help='pair distribution function')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 trajectory file')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--sample', type=int, help='phase space sample number')
    parser.add_argument('--bins', type=int, help='number of histogram bins')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.set_defaults(sample=-1, bins=100)

