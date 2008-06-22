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
Plot computation time versus system size
"""
def plot(args):
    name, variant = None, None
    data = {}
    for fn in args.input:
        f = None
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        try:
            # Lennard-Jones fluid or hardspheres simulation
            if name is None:
                name = f.root.parameters.program._v_attrs.name
            elif name != f.root.parameters.program._v_attrs.name:
                raise SystemExit('conflicting program name in file: %s' % fn)
            # program variant (e.g. +3D +CUDA +VVERLET +CELL)
            if variant is None:
                variant = f.root.parameters.program._v_attrs.variant
            elif variant != f.root.parameters.program._v_attrs.variant:
                raise SystemExit('conflicting program variant in file: %s' % fn)

            density = f.root.parameters.mdsim._v_attrs.density

            if '+CUDA' in variant:
                time = f.root.times.gpu.mdstep._v_attrs.mean
            else:
                time = f.root.times.host.mdstep._v_attrs.mean

            if not density in data:
                data[density] = {}
            if not args.loglog:
                # number of particles in thousands
                N = f.root.parameters.mdsim._v_attrs.particles / 1000
                # computation time in milliseconds
                data[density][N] = time * 1000
            else:
                # number of particles
                N = f.root.parameters.mdsim._v_attrs.particles
                # computation time in seconds
                data[density][N] = time

        except tables.exceptions.NoSuchNodeError:
            print >> sys.stderr, 'WARNING: skipping invalid file: %s' % fn

        finally:
            f.close()

    ax = plt.axes()
    ax.set_color_cycle(['m', 'b', 'c', 'g', 'r'])

    plot = args.loglog and plt.loglog or plt.plot
    for (density, set) in sorted(data.iteritems()):
        d = numpy.array(sorted(set.iteritems()))
        plot(d[:, 0], d[:, 1], '+-', label=r'$\rho^* = %.2g$' % density)

    plt.axis('tight')
    plt.legend(loc='best', labelsep=0.01, pad=0.1, axespad=0.025)
    if not args.loglog:
        plt.xlabel(r'number of particles / 1000')
        plt.ylabel(r'computation time / ms')
    else:
        plt.xlabel(r'number of particles')
        plt.ylabel(r'computation time / s')

    plt.savefig(args.output)


def add_parser(subparsers):
    parser = subparsers.add_parser(command, help='computation time versus system size')
    parser.add_argument('input', nargs='+', help='HDF5 data files')
    parser.add_argument('--output', required=True, help='output filename')
    parser.add_argument('--loglog', action='store_true', help='plot both axes with logarithmic scale')

command = 'perf'

