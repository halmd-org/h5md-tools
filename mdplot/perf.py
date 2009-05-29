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
import sys
import tables
from numpy import *


"""
Plot computation time versus system size
"""
def perf(args, ax):
    from matplotlib import pyplot as plot

    name, variant = None, None
    data = {}
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        # HDF5 root group
        H5 = f.root

        try:
            # Lennard-Jones fluid or hardspheres simulation
            if name is None:
                name = H5.param.program._v_attrs.name
            elif name != H5.param.program._v_attrs.name:
                raise SystemExit('conflicting program name in file: %s' % fn)

            # particle density
            density = float32('%.5g' % H5.param.mdsim._v_attrs.density)
            # mean MD simulation step time in equilibration phase
            time = H5.times._v_children[args.type][0][0]

            if not density in data:
                data[density] = {}
            if not args.loglog:
                # number of particles in thousands
                N = sum((H5.param.mdsim._v_attrs.particles,)) / 1000
                if args.speedup and N in data[density]:
                    # relative performance speedup
                    data[density][N] = data[density][N] / (time * 1000)
                else:
                    # computation time in milliseconds
                    data[density][N] = time * 1000
            else:
                # number of particles
                N = H5.param.mdsim._v_attrs.particles
                # computation time in seconds
                data[density][N] = time

        except tables.exceptions.NoSuchNodeError:
            print >> sys.stderr, 'WARNING: skipping invalid file: %s' % fn

        finally:
            f.close()

    plotf = args.loglog and ax.loglog or ax.plot
    for (density, set) in reversed(sorted(data.iteritems())):
        d = array(sorted(set.iteritems()))
        plotf(d[:, 0], d[:, 1], '.-', label=r'$\rho^* = %.1f$' % density)

    if not args.loglog:
        plot.setp(ax, xlabel=args.xlabel or r'number of particles / 1000')
        if args.speedup:
            plot.setp(ax, ylabel=args.ylabel or r'GPU speedup over CPU')
        else:
            plot.setp(ax, ylabel=args.ylabel or r'mean MD step time / ms')
    else:
        plot.setp(ax, xlabel=args.xlabel or r'number of particles')
        plot.setp(ax, ylabel=args.ylabel or r'mean MD step time / s')


"""
Plot composition of MD step times
"""
def perf_detail(args, ax):
    from matplotlib import pyplot as plot
    param = lambda f: sum((f.root.param.mdsim._v_attrs.particles,))
    data = {}

    steps = []
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)
        x = param(f)
        for dset in f.root.times._v_children:
            y, n = f.root.times._v_children[dset][0][::2]
            if not n > 0:
                continue
            if not dset in data:
                data[dset] = []
            data[dset].append((x, y, n))
        steps.append(f.root.times.mdstep[0][2])
        f.close()

    offset = zeros((len(args.input)))
    steps = array(steps)
    c = 0
    for counter, dset in reversed(sorted(data.iteritems())):
        times = array(dset)
        times = times[times[:, 0].argsort()]
        color = args.colors[c % len(args.colors)]
        if not counter == 'mdstep':
            x, y = times[:, 0], array(offset[:])
            offset += times[:, 1] * times[:, 2] / steps
            x = x.tolist()
            x.reverse()
            x = array(x)
            y = y.tolist()
            y.reverse()
            y = array(y)
            x, y = append(x, times[:, 0]), append(y, offset)
            ax.fill(x, y, lw=0, label=counter, facecolor=color)
        else:
            ax.plot(times[:, 0], times[:, 1], '.-', label=counter, color=color)
        c += 1

    plot.setp(ax, xlabel=args.xlabel or r'number of particles')
    plot.setp(ax, ylabel=args.ylabel or r'mean MD step time / s')


def plot(args):
    from matplotlib import pyplot as plot

    ax = plot.axes()
    ax.set_color_cycle(args.colors)

    if args.detail:
        perf_detail(args, ax)
    else:
        perf(args, ax)

    ax.axis('tight')
    if args.xlim:
        plot.setp(ax, xlim=args.xlim)
    if args.ylim:
        plot.setp(ax, ylim=args.ylim)

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('perf', help='computation time versus system size')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 performance files')
    parser.add_argument('--type', help='performance counter type')
    parser.add_argument('--detail', action='store_true', help='plot composition of MD step times')
    parser.add_argument('--speedup', action='store_true', help='compare two data sets')
    parser.add_argument('--loglog', action='store_true', help='plot both axes with logarithmic scale')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.set_defaults(type='mdstep')

