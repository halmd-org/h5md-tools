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
from re import split
import tables
import mdplot.label
import ssf
from mdplot.ext import _static_structure_factor

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as ga
from time import time

"""
Compute and plot static structure factor
"""
def plot(args):
    from matplotlib import pyplot as plt

    if args.cuda:
        make_cuda_kernels()

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
                samples = H5.trajectory._v_children[args.flavour].r
            else:
                samples = H5.trajectory.r

            # periodically extended particle positions
            # possibly read several samples
            idx = [int(x) for x in split(':', args.sample)]
            if len(idx) == 1 :
                samples = array([samples[idx[0]]])
            elif len(idx) == 2:
                samples = samples[idx[0]:idx[1]]
            # periodic simulation box length
            L = param.mdsim._v_attrs.box_length
            # number of particles
            N = sum(param.mdsim._v_attrs.particles)
            # positional coordinates dimension
            dim = param.mdsim._v_attrs.dimension

            # store attributes for later use before closing the file
            attrs = mdplot.label.attributes(param)

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

        # choose q vectors on surface of Ewald's sphere
        q_list = {}
        for j, q_val in enumerate(q_range):
            q_list[j] = q_grid[where(abs(q_norm - q_val) < q_err)]
            if args.verbose:
                print '|q| = %.2f\t%4d vectors' % (q_val, len(q_list[j]))

        # compute static structure factor over |q| range
        S_q = zeros(nq)
        S_q2 = zeros(nq)
        timer_host = 0
        timer_gpu = 0
        global timer_copy, timer_memory, timer_exp, timer_sum
        timer_copy = 0
        timer_memory = 0
        timer_exp = 0
        timer_sum = 0
        # average static structure factor over many samples
        for r in samples:
            # average over q vectors
            for j,q in q_list.items():
                if args.cuda:
                    t1 = time()
                    S_q[j] += ssf_cuda(q, r, args.block_size, j==0)
                    t2 = time()
                    timer_gpu += t2 - t1
                    if args.profiling:
                        S_q2[j] += _static_structure_factor(q, r)
                        t3 = time()
                        timer_host += t3 - t2
                else:
                    S_q[j] += _static_structure_factor(q, r)

        if args.cuda and args.profiling:
            diff = abs(S_q - S_q2) / S_q
            idx = where(diff > 1e-6)
            print diff[idx], '@', q_range[idx]

            print 'Copying: %.3f ms' % (1e3 * timer_copy)
            print 'Memory allocation: %.3f ms' % (1e3 * timer_memory)
            print 'Exponential: %.3f ms' % (1e3 * timer_exp)
            print 'Summation: %.3f ms' % (1e3 * timer_sum)
            print 'Overhead: %.3f ms' % (1e3 * (timer_gpu - \
                    (timer_copy + timer_memory + timer_exp + timer_sum)))
            print
            print 'GPU  execution time: %.3f s' % (timer_gpu)
            print 'Host execution time: %.3f s' % (timer_host)
            print 'Speedup: %.1f' % (timer_host / timer_gpu)

        S_q /= samples.shape[0]

        if args.label:
            label = args.label[i % len(args.label)] % attrs

        elif args.legend or not args.small:
            basename = os.path.splitext(os.path.basename(fn))[0]
            label = r'%s' % basename.replace('_', r'\_')

        if args.title:
            title = args.title % attrs

        c = args.colors[i % len(args.colors)]
        ax.plot(q_range, S_q, '-', color=c, label=label)
        ax.plot(q_range, S_q, 'o', markerfacecolor=c, markeredgecolor=c, markersize=2)
        if args.dump:
            f = open(args.dump, 'a')
            print >>f, '# %s' % label.replace(r'\_', '_')
            savetxt(f, array((q_range, S_q)).T)
            print >>f, '\n'

    # optionally plot power laws
    if args.power_law:
        p = reshape(args.power_law, (-1, 4))
        for (pow_exp, pow_coeff, pow_xmin, pow_xmax) in p:
            px = logspace(log10(pow_xmin), log10(pow_xmax), num=100)
            py = pow_coeff * pow(px, pow_exp)
            ax.plot(px, py, 'k--')

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

    ax.axis('tight')
    if args.xlim:
        plt.setp(ax, xlim=args.xlim)
    if args.ylim:
        plt.setp(ax, ylim=args.ylim)

    plt.xlabel(args.xlabel or r'$\lvert\textbf{q}\rvert\sigma$')
    plt.ylabel(args.ylabel or r'$S(\lvert\textbf{q}\rvert)$')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)

def make_cuda_kernels():
    from pycuda.compiler import SourceModule
    from pycuda.reduction import ReductionKernel

    global ssf_module, tex_q, sum_kernel

    # read and compile file ssf_kernel.cu
    ssf_kernel_source = file(os.path.join(mdplot.__path__[0], 'gpu/ssf_kernel.cu')).read()
    ssf_module = SourceModule(ssf_kernel_source, no_extern_c=True)

#    compute_ssf.prepare("PPP", block=(128, 1, 1))

def ssf_cuda(q, r, block_size=128, copy=True):
    nq, dim = q.shape
    npart = r.shape[0]

    global timer_copy, timer_memory, timer_zero, timer_exp, timer_sum

    # CUDA execution dimensions
    block = (block_size, 1, 1)
    grid = (60, 1)

    # access module functions, textures and constants
    if not 'compute_ssf' in globals():
        global compute_ssf, finalise_ssf, tex_q, dim_ptr, npart_ptr, nq_ptr
        compute_ssf = ssf_module.get_function('compute_ssf')
        finalise_ssf = ssf_module.get_function('finalise_ssf')
        tex_q = ssf_module.get_texref('tex_q')
        dim_ptr = ssf_module.get_global('dim')[0]
        npart_ptr = ssf_module.get_global('npart')[0]
        nq_ptr = ssf_module.get_global('nq')[0]

    # set device constants
    t1 = time()
    cuda.memset_d32(dim_ptr, dim, 1)
    cuda.memset_d32(npart_ptr, npart, 1)
    cuda.memset_d32(nq_ptr, nq, 1)
    t2 = time()
    timer_copy += t2 - t1

    # copy particle positions to device
    # (x0, x1, x2, ..., xN, y0, y1, y2, ..., yN, z0, z1, z2, ..., zN)
    if copy:
        global gpu_r
        t1 = time()
        gpu_r = ga.to_gpu(r.T.flatten().astype(float32))
        t2 = time()
        timer_copy += t2 - t1

    # allocate space for results
    t1 = time()
    gpu_sin = ga.empty(int(nq * prod(grid)), float32)
    gpu_cos = ga.empty(int(nq * prod(grid)), float32)
    gpu_ssf = ga.empty(int(prod(grid)), float32)
    t2 = time()
    timer_memory += t2 - t1

    # copy group of wavevectors with (almost) equal magnitude
    t1 = time()
    gpu_q = ga.to_gpu(q.flatten().astype(float32))
    gpu_q.bind_to_texref_ext(tex_q)
    t2 = time()
    timer_copy += t2 - t1

    # compute exp(iq·r) for each particle
    t1 = time()
    compute_ssf(gpu_sin, gpu_cos, gpu_r,
                block=block, grid=grid, texrefs=[tex_q])
    t2 = time()
    # compute sum(sin(q·r))^2 + sum(cos(q·r))^2 per wavevector
    # and sum over wavevectors
    finalise_ssf(gpu_sin, gpu_cos, gpu_ssf, int32(prod(grid)),
                 block=block, grid=grid)
    result = sum(gpu_ssf.get())
    t3 = time()
    timer_exp += t2 - t1
    timer_sum += t3 - t2

    # normalize result with #wavevectors and #particles
    return result / (nq * npart)

def add_parser(subparsers):
    parser = subparsers.add_parser('ssf', help='static structure factor')
    parser.add_argument('input', nargs='+', metavar='INPUT', help='HDF5 trajectory file')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--sample', help='index of phase space sample(s)')
    parser.add_argument('--q-limit', type=float, help='maximum value of |q|')
    parser.add_argument('--q-error', type=float, help='relative deviation of |q|')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--power-law', type=float, nargs='+', help='plot power law curve(s)')
    parser.add_argument('--cuda', action='store_true', help='use CUDA device to speed up the computation')
    parser.add_argument('--block-size', type=int, help='block size to be used for CUDA calls')
    parser.add_argument('--profiling', action='store_true', help='output profiling results and compare with host version')
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(sample='0', q_limit=25, q_error=0.1, block_size=256)

