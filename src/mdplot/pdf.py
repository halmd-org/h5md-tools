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
import sys
import tables


"""
Plot pair distribution function
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
        n = f.root.parameters.mdsim._v_attrs.particles
        # positional coordinates dimension
        dim = f.root.parameters.mdsim._v_attrs.dimension

        rij = numpy.zeros([n * (n - 1) / 2])
        for (i, j) in enumerate(range(n - 1, 0, -1)):
            k = i * (n + j) / 2
            # particle distance vector
            dr = r[:j] - r[i + 1:]
            # minimum image distance
            dr = dr - numpy.round(dr / box) * box
            # squared particle distance
            rr = dr[:, 0] * dr[:, 0] + dr[:, 1] * dr[:, 1]
            if dim == 3:
                rr = rr + dr[:, 2] * dr[:, 2]
            # absolute particle distance
            rij[k:(k + j)] = numpy.sqrt(rr)

    except tables.exceptions.NoSuchNodeError:
        raise SystemExit('missing simulation data in file: %s' % args.input)

    finally:
        f.close()

    # compute pair correlation function
    g, bins = numpy.histogram(rij, bins=args.bins, normed=True, new=True)

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

