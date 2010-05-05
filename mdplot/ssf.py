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
from matplotlib import ticker
from numpy import *
import sys
import tables
import mdplot.label
import ssf
from mdplot.ext import _static_structure_factor

"""
Plot static structure factor
"""
def plot(args):
    from matplotlib import pyplot as plt

    ax = plt.axes()
    label = None
    ax.axhline(y=1, color='black', lw=0.5)
    ax.set_color_cycle(args.colors)

    for (i, fn) in enumerate(args.input):
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        param = H5.param
        try:
            if args.flavour:
                trajectory = H5.trajectory._v_children[args.flavour]
            else:
                trajectory = H5.trajectory

            # periodically extended particle positions
            r = trajectory.r[args.sample]
            # periodic simulation box length
            L = param.mdsim._v_attrs.box_length
            # number of particles
            N = sum(param.mdsim._v_attrs.particles)
            # positional coordinates dimension
            dim = param.mdsim._v_attrs.dimension

        except IndexError:
            raise SystemExit('invalid phase space sample offset')
        except tables.exceptions.NoSuchNodeError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)
        finally:
            f.close()

        # reciprocal lattice distance
        q_min = (2 * pi / L)
        # number of values for |q|
        nq = int(args.q_limit / q_min)
        # absolute deviation of |q|
        q_err = q_min * args.q_error

        # generate n-dimensional q-grid
        q_grid = q_min * squeeze(dstack(reshape(indices(repeat(nq + 1, dim)), (dim, -1))))
        # compute absolute |q| values of q-grid
        q_norm = sqrt(sum(q_grid * q_grid, axis=1))

        # |q| value range
        q_range = q_min * arange(1, nq + 1)

        # compute static structure factor over |q| range
        S_q = zeros(nq)
        for i, q_val in enumerate(q_range):
            # choose q vectors on surface of Ewald's sphere
            q = q_grid[where(abs(q_norm - q_val) < q_err)]
            if args.verbose:
                print '|q| = %.2f\t%4d vectors' % (q_val, len(q))
            # average static structure factor over q vectors
            S_q[i] = _static_structure_factor(q, r)

        if args.label:
            label = args.label[i % len(args.label)] % mdplot.label.attributes(param)

        elif args.legend or not args.small:
            basename = os.path.splitext(os.path.basename(fn))[0]
            label = r'%s' % basename.replace('_', r'\_')

        if args.title:
            title = args.title % mdplot.label.attributes(param)

        ax.plot(q_range, S_q, label=label)

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    plt.xlabel(args.xlabel or r'$\lvert\textbf{q}\rvert\sigma$')
    plt.ylabel(args.ylabel or r'$S(\lvert\textbf{q}\rvert)$')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('ssf', help='static structure factor')
    parser.add_argument('input', nargs='+', metavar='INPUT', help='HDF5 trajectory file')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--sample', type=int, help='phase space sample offset')
    parser.add_argument('--q-limit', type=float, help='maximum value of |q|')
    parser.add_argument('--q-error', type=float, help='relative deviation of |q|')
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(sample=0, q_limit=25, q_error=0.1)

