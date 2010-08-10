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

#define BLOCK_SIZE 32

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

// global constants
__constant__ int npart;
__constant__ int dim;

// compute exp(i q·r) for a single particle,
// return block sum of results
__global__ void compute_ssf(float* sin_block, float* cos_block, float const* r, int offset)
{
    __shared__ float sin_[BLOCK_SIZE];
    __shared__ float cos_[BLOCK_SIZE];

    const int i = GTID;
    if (i >= npart)
        return;

    float q_r = 0;
    for (int k=0; k < dim; k++) {
        q_r += tex1Dfetch(tex_q, offset * dim + k) * r[i + k * npart];
    }
    sin_[TID] = sin(q_r);
    cos_[TID] = cos(q_r);
    __syncthreads();

    // accumulate results within block
    #if (BLOCK_SIZE >= 512)
        if (TID < 256) {
            sin_[TID] += sin_[TID + 256];
            cos_[TID] += cos_[TID + 256];
        }
        __syncthreads();
    #endif

    #if (BLOCK_SIZE >= 256)
        if (TID < 128) {
            sin_[TID] += sin_[TID + 128];
            cos_[TID] += cos_[TID + 128];
        }
        __syncthreads();
    #endif

    #if (BLOCK_SIZE >= 128)
        if (TID < 64) {
            sin_[TID] += sin_[TID + 64];
            cos_[TID] += cos_[TID + 64];
        }
        __syncthreads();
    #endif

    if (TID < 32) {
        if (BLOCK_SIZE >= 64) {
            sin_[TID] += sin_[TID + 32];
            cos_[TID] += cos_[TID + 32];
        }
        if (BLOCK_SIZE >= 32) {
            sin_[TID] += sin_[TID + 16];
            cos_[TID] += cos_[TID + 16];
        }
        if (BLOCK_SIZE >= 16) {
            sin_[TID] += sin_[TID + 8];
            cos_[TID] += cos_[TID + 8];
        }
        if (BLOCK_SIZE >= 8) {
            sin_[TID] += sin_[TID + 4];
            cos_[TID] += cos_[TID + 4];
        }
        if (BLOCK_SIZE >= 4) {
            sin_[TID] += sin_[TID + 2];
            cos_[TID] += cos_[TID + 2];
        }
        if (BLOCK_SIZE >= 2) {
            sin_[TID] += sin_[TID + 1];
            cos_[TID] += cos_[TID + 1];
        }
    }

    if (TID < 1) {
        sin_block[BID] = sin_[0];
        cos_block[BID] = cos_[0];
    }
}

