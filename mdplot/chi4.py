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


"""
Plot four-point susceptibility
"""
def plot(args):
    from matplotlib import pyplot as plt

    ax = plt.axes()
    label = None
    # plot zero line
    ax.axhline(y=0, color='black', lw=0.5)

    i = 0
    for fn in args.input:
        try:
            f = tables.openFile(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f.root
        param = H5.param
        try:
            if args.flavour:
                H5 = H5._v_children[args.flavour]

            version = param.program._v_attrs.version
            # number of q-values
            if version < 'v0.2.5.2-2-g92f02d5':
                nq = param.correlation._v_attrs.q_values
                if not isscalar(nq):
                    nq = len(nq)
            else:
                nq = len(param.correlation._v_attrs.q_values)
            # number of particles
            npart = param.mdsim._v_attrs.particles
            if not isscalar(npart):
                if args.flavour == 'AA':
                    npart = npart[0]
                elif args.flavour == 'BB':
                    npart = npart[1]
                else:
                    raise SystemExit('Don\'t know how to handle mixture, npart is not a scalar (FIXME)')

            # merge block levels, discarding time zero
            sisf = H5._v_children['SISF'][:, :, 1:, :]
            sisf.shape = nq, -1, sisf.shape[-1]
            sisf2 = H5._v_children['SISF2'][:, :, 1:, :]
            sisf2.shape = nq, -1, sisf2.shape[-1]

            # q-vector magnitudes
            q_values = sorted(unique(sisf[:, :, 0]))
            if args.q_values:
                # generate intervals with q-vectors in interval centers
                q_bins = (q_values + append((0, ), q_values[:-1])) / 2
                # choose nearest neighbour q-vectors
                q_values = choose(digitize(args.q_values, q_bins) - 1, q_values)

            for q in q_values:
                # select blocks with matching q-vector
                d = sisf.compress(sisf[:, 0, 0] == q, axis=0)
                d2 = sisf2.compress(sisf2[:, 0, 0] == q, axis=0)
                # time-order correlation function samples
                time_order = d[0, :, 1].argsort()
                x = d[0, time_order, 1]
                y1 = d[0, time_order, 2]
                y2 = d2[0, time_order, 2]
                # compute four-point susceptibility
                y = npart * (y2 - y1 * y1)

                if not len(x):
                    raise SystemExit('empty plot range')

                if args.label:
                    attrs = mdplot.label.attributes(param)
                    attrs['q'] = r'%.2f' % q
                    label = args.label[i % len(args.label)] % attrs
                elif args.legend or not args.small:
                    basen = os.path.splitext(os.path.basename(fn))[0]
                    label = r'%s: $q = %.2f$' % (basen.replace('_', r'\_'), q)

                # cycle plot styles
                fmt = args.colors[i % len(args.colors)]
                i += 1

                if isinstance(fmt, str):
                    # arbitrary plot style
                    ax.plot(x, y, fmt=fmt, label=label)
                else:
                    # rgb color tuple
                    ax.plot(x, y, color=fmt, label=label)

        except tables.exceptions.NoSuchNodeError:
            raise SystemExit('missing simulation data in file: %s' % fn)

        finally:
            f.close()

    if args.axes == 'xlog':
        ax.set_xscale('log')
    elif args.axes == 'ylog':
        ax.set_yscale('log')
    elif args.axes == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')

    if args.legend or not args.small:
        l = ax.legend(loc=args.legend)
        l.legendPatch.set_alpha(0.7)

    plt.axis('tight')
    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    plt.xlabel(args.xlabel or r'$t^*$')
    plt.ylabel(args.ylabel or r'$N \Big(\langle f_s(q, t^*)^2 \rangle - \langle f_s(q, t^*) \rangle^2 \Big)$')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)


def add_parser(subparsers):
    parser = subparsers.add_parser('chi4', help='four-point susceptibility')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='HDF5 correlations file')
    parser.add_argument('--flavour', help='flavour of correlation functions, selects subgroup in HDF5 file')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--q-values', type=float, nargs='+', help='q-vector magnitude(s)')

