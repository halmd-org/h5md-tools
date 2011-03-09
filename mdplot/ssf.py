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
from time import time

"""
Compute and plot static structure factor
"""
def plot(args):
    from matplotlib import pyplot as plt

    # import pycuda only if required
    if args.cuda:
        import pycuda.autoinit
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

        try:
            H5 = f.root
            param = H5.param

            # determine file type, prefer precomputed SSF data
            if 'structure' in H5._v_groups and 'ssf' in H5.structure._v_groups:
                # load SSF from file
                H5ssf = H5.structure.ssf._v_children[args.flavour or 'AA']
                q = H5.structure.ssf.wavenumber[0]
                S_q, S_q_err = load_ssf(H5ssf, args)

            elif 'trajectory' in H5._v_groups:
                # compute SSF from trajectory data
                if args.flavour:
                    H5trj = H5.trajectory._v_children[args.flavour]
                else:
                    H5trj = H5.trajectory

                q, S_q = ssf_from_trajectory(H5trj, param, args)
            else:
                raise SystemExit('Input file provides neither SSF data nor a trajectory')

            # before closing the file, store attributes for later use
            attrs = mdplot.label.attributes(param)

        except IndexError:
            raise SystemExit('invalid phase space sample offset')
        except tables.exceptions.NoSuchNodeError as what:
            raise SystemExit(str(what) + '\nmissing simulation data in file: %s' % fn)
        finally:
            f.close()

        if args.label:
            label = args.label[i % len(args.label)] % attrs

        elif args.legend or not args.small:
            basename = os.path.splitext(os.path.basename(fn))[0]
            label = r'%s' % basename.replace('_', r'\_')

        if args.title:
            title = args.title % attrs

        c = args.colors[i % len(args.colors)]
        ax.plot(q, S_q, '-', color=c, label=label)
        if 'S_q_err' in locals():
            ax.errorbar(q, S_q, S_q_err, fmt='o', color=c, markerfacecolor=c, markeredgecolor=c, markersize=2, linewidth=.5)
        else:
            ax.plot(q, S_q, 'o', markerfacecolor=c, markeredgecolor=c, markersize=2)

        # optionally plot compressibility
        if args.compressibility and attrs['density'] and attrs['temperature']:
            chi_T = args.compressibility[i % len(args.compressibility)]
            rho = attrs['density']
            temp = attrs['temperature']
            x = linspace(1e-3, 3 * q[0], num=2)
            y = empty_like(x)
            y.fill(chi_T * rho * temp)
            ax.plot(x, y, ':', color=c, linewidth=.5)

        # write plot data to file
        if args.dump:
            f = open(args.dump, 'a')
            print >>f, '# %s' % label.replace(r'\_', '_')
            if 'S_q_err' in locals():
                print >>f, '# q   S_q   S_q_err'
                savetxt(f, array((q, S_q, S_q_err)).T)
            else:
                print >>f, '# q   S_q'
                savetxt(f, array((q, S_q)).T)
            print >>f, '\n'
            f.close()

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
    else:
        plt.setp(ax, ylim=(0, plt.ylim()[1]))

    plt.xlabel(args.xlabel or r'$\lvert\textbf{q}\rvert\sigma$')
    plt.ylabel(args.ylabel or r'$S(\lvert\textbf{q}\rvert)$')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, dpi=args.dpi)

"""
Load precomputed SSF data from HDF5 file
"""
def load_ssf(H5data, args):
    idx = [int(x) for x in split(':', args.sample)]
    if len(idx) == 1:
        idx = idx + [-1]
    ssf = H5data[idx[0]:idx[1]]

    # compute mean
    S_q = mean(ssf[..., 0], axis=0)

    # error has two contributions:
    # sum squares of individual errors
    S_q_err = sum(pow(ssf[..., 1], 2), axis=0) / pow(ssf.shape[0], 2)
    # estimate error from blocking
    nblocks = min(ssf.shape[0], 10)
    if nblocks > 1:
        # cut size of data to a multiple of args.blocks
        ssf = ssf[:int(nblocks * floor(ssf.shape[0] / nblocks))]
        # reshape in equally sized blocks
        ssf = reshape(ssf, (nblocks, -1,) + ssf.shape[1:])
        S_q_err += var(mean(ssf[..., 0], axis=1), axis=0) / (nblocks - 1)
    S_q_err = sqrt(S_q_err)

    return S_q, S_q_err

"""
Compute static structure factor from trajectory data
"""
def ssf_from_trajectory(H5data, param, args):
    # backwards compatibility
    compatibility = 'file_version' not in param._v_attrs
    if not compatibility:
        samples = H5data.position
    else:
        print 'compatibility mode'
        samples = H5data.r

    # read periodically extended particle positions,
    # read one or several samples, convert to single precision
    idx = [int(x) for x in split(':', args.sample)]
    if len(idx) == 1:
        samples = array([samples[idx[0]]], dtype=float32)
    elif len(idx) == 2:
        samples = array(samples[idx[0]:idx[1]], dtype=float32)

    if not compatibility:
        # positional coordinates dimension
        dim = param.box._v_attrs.dimension
        # periodic simulation box length
        L = param.box._v_attrs.length
        # number of particles
        N = sum(param.box._v_attrs.particles)
    else:
        # positional coordinates dimension
        dim = param.mdsim._v_attrs.dimension
        # edge lengths of cubic simulation box
        L = repeat(param.mdsim._v_attrs.box_length, dim)
        # number of particles
        N = sum(param.mdsim._v_attrs.particles)

    # unit cell (basis vectors) of reciprocal lattice
    q_basis = float32(2 * pi / L)
    # minimal wavenumber
    q_min = max(q_basis)
    # number of values for |q|
    nq = int(args.q_limit / q_min)
    # absolute deviation of |q|
    q_err = q_min * args.q_error

    # generate n-dimensional q-grid
    q_grid = q_basis * squeeze(dstack(
                reshape(
                    indices(repeat(nq + 1, dim), dtype=float32)
                  , (dim, -1)
                )))

    # compute absolute |q| values of q-grid
    q_norm = sqrt(sum(q_grid * q_grid, axis=1))

    # choose q vectors on surface of Ewald's sphere
    # with magnitudes from linearly spaced grid
    q_range = []
    q_list = []
    for q_val in q_min * arange(1, nq + 1):
        q_ = q_grid[where(abs(q_norm - q_val) < q_err)]
        if len(q_) > 0:
            j = len(q_list)
            q_list.append(q_)
            q_range.append(q_val)
            if args.verbose:
                print '|q| = %.2f\t%4d vectors' % (q_val, len(q_))
    # adjust nq to actual number of wavenumbers
    nq = len(q_range)

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
        for j,q in enumerate(q_list):
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

    return q_range, S_q

def make_cuda_kernels():
    from pycuda.compiler import SourceModule
    from pycuda.reduction import ReductionKernel

    global ssf_module, tex_q, sum_kernel

    # read and compile file ssf_kernel.cu
    ssf_kernel_source = file(os.path.join(mdplot.__path__[0], 'gpu/ssf_kernel.cu')).read()
    ssf_module = SourceModule(ssf_kernel_source, no_extern_c=True)

#    compute_ssf.prepare("PPP", block=(128, 1, 1))

def ssf_cuda(q, r, block_size=128, copy=True):
    import pycuda.driver as cuda
    import pycuda.gpuarray as ga

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
    parser.add_argument('input', nargs='+', metavar='INPUT', help='HDF5 file with trajectory or ssf data')
    parser.add_argument('--flavour', help='particle flavour')
    parser.add_argument('--sample', help='index of phase space sample(s)')
    parser.add_argument('--q-limit', type=float, help='maximum value of |q|')
    parser.add_argument('--q-error', type=float, help='relative deviation of |q|')
    parser.add_argument('--xlim', metavar='VALUE', type=float, nargs=2, help='limit x-axis to given range')
    parser.add_argument('--ylim', metavar='VALUE', type=float, nargs=2, help='limit y-axis to given range')
    parser.add_argument('--axes', choices=['xlog', 'ylog', 'loglog'], help='logarithmic scaling')
    parser.add_argument('--power-law', type=float, nargs='+', help='plot power law curve(s)')
    parser.add_argument('--compressibility', type=float, nargs='+', help='isothermal compressibility for S(q → 0)')
    parser.add_argument('--cuda', action='store_true', help='use CUDA device to speed up the computation')
    parser.add_argument('--block-size', type=int, help='block size to be used for CUDA calls')
    parser.add_argument('--profiling', action='store_true', help='output profiling results and compare with host version')
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(sample='0', q_limit=15, q_error=0.1, block_size=256)

