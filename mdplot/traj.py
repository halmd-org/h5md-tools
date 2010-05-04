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
import glob
from matplotlib import ticker
import numpy
import subprocess
import sys
import tables


"""
Plot trajectory projections on a 2-dimensional plane
"""
def plot(args):
    from matplotlib import pyplot as plt

    try:
        f = tables.openFile(args.input, mode='r')
    except IOError:
        raise SystemExit('failed to open HDF5 file: %s' % args.input)

    # HDF5 root group
    H5 = f.root

    try:
        if args.flavour:
            trajectory = H5.trajectory._v_children[args.flavour]
        else:
            trajectory = H5.trajectory

        # periodic simulation box length
        box = H5.param.mdsim._v_attrs.box_length
        # positional coordinates dimension
        dim = H5.param.mdsim._v_attrs.dimension

        major_formatter = ticker.FormatStrFormatter('%.3g')
        if 'cell_length' in H5.param.mdsim._v_attrs:
            # set tick interval to cell edge length
            major_locator = ticker.MultipleLocator(H5.param.mdsim._v_attrs.cell_length)
            grid = True
        else:
            major_locator = ticker.MultipleLocator(box / 10)
            grid = False

        if 'pair_separation' in H5.param.mdsim._v_attrs:
            # hardsphere particle diameter
            diameter = H5.param.mdsim._v_attrs.pair_separation
        else:
            # repulsive part of Lennard-Jones potential
            diameter = numpy.power(2, 1./6.)

        if args.output is None:
            # use input basename as default output basename
            fn, ext = os.path.splitext(args.input)
            mfn = '%s.ogg' % fn
        else:
            fn, ext = os.path.splitext(args.output)
            mfn = args.output

        # remove stray plot files
        for g in glob.glob('%s_*.png' % fn):
            os.unlink(g)

        fig = plt.figure(1, figsize=(8, 8))
        figscale = fig.dpi * (1.0 / 72.0)

        # phase space sample subselection
        s = slice(args.start, args.count and (args.start + args.stride * args.count) or None, args.stride)

        sys.stdout.write('plot: %6sf' % '')
        for (i, (r, t)) in enumerate(zip(trajectory.r[s], H5.trajectory.t[s])):
            # enforce periodic boundary conditions
            r[:] = r[:] - numpy.floor(r[:] / box) * box

            ax = plt.axes((0.06, 0.06, 0.88, 0.88))
            axscale = ax.get_window_extent().width / box
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_major_formatter(major_formatter)
            ax.yaxis.set_major_locator(major_locator)
            ax.yaxis.set_major_formatter(major_formatter)
            ax.grid(grid)
            # scale particle diameter from data units to points
            d = diameter * axscale / figscale
            if dim == 3:
                cmap = color_wheel(r[:, 2] / box)
                plot = lambda x, y: ax.scatter(x, y, s=(d * d), c=cmap, edgecolors='none', alpha=0.9)
            else:
                plot = lambda x, y: ax.plot(x, y, 'o', markersize=d, markerfacecolor='b', markeredgecolor='b', alpha=0.5)
            # plot projections of particles in simulation box
            plot(r[:, 0], r[:, 1])
            # plot periodic particle images in neighbour boxes
            plot(r[:, 0], r[:, 1] + box)
            plot(r[:, 0], r[:, 1] - box)
            plot(r[:, 0] + box, r[:, 1] + box)
            plot(r[:, 0] + box, r[:, 1])
            plot(r[:, 0] + box, r[:, 1] - box)
            plot(r[:, 0] - box, r[:, 1] + box)
            plot(r[:, 0] - box, r[:, 1])
            plot(r[:, 0] - box, r[:, 1] - box)
            # limit view to simulation box boundaries
            ax.axis((0, box, 0, box))
            # print simulation time
            ax.text(0.85, 0.05, r'$t^* = %.2g$' % t, transform=ax.transAxes)

            plt.savefig('%s_%06d.png' % (fn, i), dpi=args.dpi)
            plt.clf()

            # erase previously printed frame number characters
            sys.stdout.write('\010 \010' * 7)
            sys.stdout.write('%6df' % (i + 1))
            sys.stdout.flush()

    except tables.exceptions.NoSuchNodeError as what:
        raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % args.input)

    finally:
        f.close()

    sys.stdout.write('\n')
    sys.stdout.flush()

    # render movie with ffmpeg2theora
    ffmpeg = subprocess.Popen(['ffmpeg2theora', '%s_%%06d.png' % fn, '-x', '960', '-y', '960', '-S', '0', '-o', mfn, '--nosound'], close_fds=True)
    ret = ffmpeg.wait()
    if ret:
        raise SystemExit('mencoder exited with error code %d' % ret)

    # remove plot files
    for g in glob.glob('%s_*.png' % fn):
        os.unlink(g)


"""
Convert array of uniform numbers to RGB tuples from color wheel
"""
def color_wheel(x):
    triang = lambda x: (2 - numpy.abs(x)).clip(0.0, 1.0)
    x = x * 6
    r = triang(x - 2)
    g = triang(x - 4)
    b = triang(x - 6) + triang(x)
    return zip(r, g, b)


def add_parser(subparsers):
    parser = subparsers.add_parser('traj', help='trajectory projections on a 2-dimensional plane')
    parser.add_argument('input', metavar='INPUT', help='HDF5 trajectory file')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--start', type=int, help='phase space sample offset')
    parser.add_argument('--count', type=int, help='phase space sample count')
    parser.add_argument('--stride', type=int, help='phase space sample stride')
    parser.set_defaults(start=0, count=0, stride=1)

