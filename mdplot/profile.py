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
        # backwards compatibility
        compatibility = 'file_version' not in H5.param._v_attrs
        if compatibility:
            print 'compatibility mode'

        try:
            try:
                # particle positions of phase space sample
                # possibly read several samples
                trajectory = H5.trajectory
                if args.flavour:
                    trajectory = trajectory._v_children[args.flavour]
                if not compatibility:
                    position = H5.trajectory.position
                else:
                    position = H5.trajectory.r

                idx = [int(x) for x in args.sample.split(':')]
                if len(idx) == 1 :
                    samples = array([position[idx[0]]])
                elif len(idx) == 2:
                    samples = position[idx[0]:idx[1]]
            except IndexError:
                raise SystemExit('out-of-bounds phase space sample number')

            if not compatibility:
                # positional coordinates dimension
                dim = H5.param.box._v_attrs.dimension
                # particle density
                density = H5.param.box._v_attrs.density
                # periodic simulation box length
                box = H5.param.box._v_attrs.length
                # number of particles
                N = H5.param.box._v_attrs.particles
                if args.flavour:
                    N = N[ord(args.flavour[:1]) - ord('A')]
                else:
                    N = N[0]
            else:
                # positional coordinates dimension
                dim = H5.param.mdsim._v_attrs.dimension
                # particle density
                density = H5.param.mdsim._v_attrs.density
                # periodic simulation box length
                box = H5.param.mdsim._v_attrs.box_length
                box = zeros(dim) + box
                # number of particles
                N = H5.param.mdsim._v_attrs.particles
                if not isscalar(N):
                    N = N[ord(args.flavour[:1]) - ord('A')]

            cutoff = args.xlim or (0, box[args.axis])
            # minimum image distances
            x = samples[..., args.axis]
            x = x - floor(x / box[args.axis]) * box[args.axis]
            histo, bins = histogram(x.flatten(), bins=args.bins, range=cutoff, new=True)
            # normalisation with volume of slabs (=bins) and number of samples,
            # the result is a local number density
            histo = array(histo, dtype=float64) / (diff(bins) * prod(box) / box[args.axis]) / samples.shape[0]

            if args.label:
                label = args.label[k % len(args.label)] % mdplot.label.attributes(H5.param)
            elif args.legend or not args.small:
                basen = os.path.splitext(os.path.basename(fn))[0]
                label = basen.replace('_', r'\_')

        except tables.exceptions.NoSuchNodeError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)

        finally:
            f.close()

        ax.plot(bins[:-1], histo, marker='.', label=label)
        if args.dump:
            f = open(args.dump, 'a')
            print >>f, '# %s' % label.replace(r'\_', '_')
            savetxt(f, array((bins[:-1], histo)).T)
            print >>f, '\n'

    ax.axis('tight')
    if args.xlim:
        plot.setp(ax, xlim=args.xlim)
    if args.ylim:
        plot.setp(ax, ylim=args.ylim)
    else:
        plot.setp(ax, ylim=(0, 1.3 * max(histo)))

    plot.setp(ax, xlabel=args.xlabel or r'$x / \sigma$')
    plot.setp(ax, ylabel=args.ylabel or r'density profile $n(x)$')
    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    if args.output is None:
        plot.show()
    else:
        plot.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('profile', help='density profile')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 trajectory file')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--sample', help='index or range of phase space samples')
    parser.add_argument('--axis', type=int, help='profile axis')
    parser.add_argument('--bins', type=int, help='number of histogram bins')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.set_defaults(
        sample='-1',
        axis=2,
        bins=50,
        )

