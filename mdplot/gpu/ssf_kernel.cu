/* mdplot - Molecular Dynamics simulation plotter
 *
 * Copyright © 2008-2010  Peter Colberg, Felix Höfling
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
 */

// thread ID within block
#define TID     threadIdx.x
// number of threads per block
#define TDIM    blockDim.x
// block ID within grid
#define BID     (blockIdx.y * gridDim.x + blockIdx.x)
// number of blocks within grid
#define BDIM    (gridDim.y * gridDim.x)
// thread ID within grid
#define GTID    (BID * TDIM + TID)
// number of threads per grid
#define GTDIM   (BDIM * TDIM)

// store q vectors in texture
texture<float, 1> tex_q;

// compute exp(i q·r) for a single particle
__global__ void compute_ssf(float *sin_, float *cos_, float *r,
                            int offset, int npart, int dim)
{
    const int i = GTID;
    if (i >= npart)
        return;

    float q_r = 0;
    for (int k=0; k < dim; k++) {
        q_r += tex1Dfetch(tex_q, offset * dim + k) * r[i + k * npart];
    }
    sin_[i] = sin(q_r);
    cos_[i] = cos(q_r);
}

