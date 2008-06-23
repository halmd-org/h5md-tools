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
import matplotlib.pyplot as plt
import numpy
from scipy.special import gamma
import sys
import tables


"""
Plot pair distribution function

M.P. Allen and D.J. Tildesley,
Computer Simulation of Liquids, 1989,
Oxford University Press, pp. 55, 183-184
"""
def plot(args):
    f = None
    try:
        f = tables.openFile(args.input, mode='r')
    except IOError:
        raise SystemExit('failed to open HDF5 file: %s' % args.input)

    try:
        try:
            # particle positions of phase space sample
            r = f.root.trajectory.positions[args.sample]
            # simulation time
            time = f.root.trajectory.time[args.sample]
        except IndexError:
            raise SystemExit('out-of-bounds phase space sample number')

        # periodic simulation box length
        box = f.root.parameters.mdsim._v_attrs.box_length
        # number of particles
        N = f.root.parameters.mdsim._v_attrs.particles
        # positional coordinates dimension
        dim = f.root.parameters.mdsim._v_attrs.dimension
        # particle density
        density = f.root.parameters.mdsim._v_attrs.density

        H = numpy.zeros(args.bins)
        for (i, j) in enumerate(range(N - 1, 0, -1)):
            # particle distance vectors
            dr = r[:j] - r[i + 1:]
            # minimum image distances
            dr = dr - numpy.round(dr / box) * box
            # squared minimum image distances
            rr = dr[:, 0] * dr[:, 0] + dr[:, 1] * dr[:, 1]
            if dim == 3:
                rr = rr + dr[:, 2] * dr[:, 2]
            # accumulate histogram of minimum image distances
            h, bins = numpy.histogram(numpy.sqrt(rr), bins=args.bins, range=(0, box), new=True)
            H = H + 2 * h

    except tables.exceptions.NoSuchNodeError:
        raise SystemExit('missing simulation data in file: %s' % args.input)

    finally:
        f.close()

    # volume of n-dimensional unit sphere
    Vn = numpy.power(numpy.pi, dim / 2.) / gamma(dim / 2. + 1.)
    # average number of atoms in ideal gas per interval
    n = Vn * density * (numpy.power(bins[1:], dim) - numpy.power(bins[:-1], dim))
    # compute pair correlation function
    g = H / n / time / N

    plt.plot(bins[:-1], g, color='m')
    plt.axis([0, box, 0, max(g)])
    plt.xlabel(r'particle distance $|\mathbf{r}_{ij}^*|$')
    plt.ylabel(r'pair distribution function $g(|\mathbf{r}_{ij}^*|)$')
    plt.savefig(args.output)


def add_parser(subparsers):
    parser = subparsers.add_parser(command, help='pair distribution function')
    parser.add_argument('input', help='HDF5 data file')
    parser.add_argument('--sample', type=int, help='phase space sample number')
    parser.add_argument('--bins', type=int, help='number of histogram bins')
    parser.add_argument('--output', required=True, help='output filename')
    parser.set_defaults(sample=-1, bins=1000)

command = 'pdf'

