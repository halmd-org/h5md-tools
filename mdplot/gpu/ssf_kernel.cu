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

#define MAX_BLOCK_SIZE 512

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

// copy enable_if_c and disable_if_c from Boost.Utility
// to avoid dependency on Boost headers
template <bool B, class T = void>
struct enable_if_c {
    typedef T type;
};

template <class T>
struct enable_if_c<false, T> {};

template <bool B, class T = void>
struct disable_if_c {
    typedef T type;
};

template <class T>
struct disable_if_c<true, T> {};

// recursive reduction function,
// terminate for threads=0
template <unsigned threads, typename T>
__device__ typename disable_if_c<threads>::type
sum_reduce(T*, T*) {}

// reduce two array simultaneously by summation,
// size of a,b must be at least 2 * threads
template <unsigned threads, typename T>
__device__ typename enable_if_c<threads>::type
sum_reduce(T* a, T* b)
{
    if (TID < threads) {
        a[TID] += a[TID + threads];
        b[TID] += b[TID + threads];
    }
    if (threads >= warpSize) {
        __syncthreads();
    }

    if (threads >= 2) {
        sum_reduce<threads / 2>(a, b);
    }
}

/* FIXME
typedef void (*sum_reduce_type)(float*, float*);
__device__ sum_reduce_type sum_reduce_select[] = {
    &sum_reduce<0>, &sum_reduce<1>, &sum_reduce<2>, &sum_reduce<4>,
    &sum_reduce<8>, &sum_reduce<16>, &sum_reduce<32>, &sum_reduce<64>,
    &sum_reduce<128>, &sum_reduce<256>
};
*/

extern "C" {

// compute exp(i q·r) for a single particle,
// return block sum of results
__global__ void compute_ssf(float* sin_block, float* cos_block, float const* r, int offset)
{
    __shared__ float sin_[MAX_BLOCK_SIZE];
    __shared__ float cos_[MAX_BLOCK_SIZE];

    const int i = GTID;
    if (i >= npart) {
        // the placeholders contribute to the block sums below,
        // set them to zero here
        sin_[TID] = 0;
        cos_[TID] = 0;
        return;
    }

    float q_r = 0;
    for (int k=0; k < dim; k++) {
        q_r += tex1Dfetch(tex_q, offset * dim + k) * r[i + k * npart];
    }
    sin_[TID] = sin(q_r);
    cos_[TID] = cos(q_r);
    __syncthreads();

    // accumulate results within block
    if (TDIM == 512) sum_reduce<256>(sin_, cos_);
    else if (TDIM == 256) sum_reduce<128>(sin_, cos_);
    else if (TDIM == 128) sum_reduce<64>(sin_, cos_);
    else if (TDIM == 64) sum_reduce<32>(sin_, cos_);
    else if (TDIM == 32) sum_reduce<16>(sin_, cos_);
    else if (TDIM == 16) sum_reduce<8>(sin_, cos_);
    else if (TDIM == 8) sum_reduce<4>(sin_, cos_);

    if (TID == 0) {
        sin_block[BID] = sin_[0];
        cos_block[BID] = cos_[0];
    }
}

// reduce two arrays by summation using a single block of threads,
// thus, all communication is based on shared memory
__global__ void sum_reduce_block(float const* a, float const* b, float *result, uint size)
{
    __shared__ float sum_a[MAX_BLOCK_SIZE];
    __shared__ float sum_b[MAX_BLOCK_SIZE];

//    assert(BDIM == 1);

    // initialise placeholders
    // if array is smaller than block size
    if (TID >= size) {
        sum_a[TID] = 0;
        sum_b[TID] = 0;
    }
    else {
        // load values from global device memory
        float acc_a = 0;
        float acc_b = 0;
        for (uint k = TID; k < size; k += TDIM) {
            acc_a += a[k];
            acc_b += b[k];
        }
        // reduced value for this thread
        sum_a[TID] = acc_a;
        sum_b[TID] = acc_b;
    }
    __syncthreads();

    // compute reduced value for all threads in block
    if (TDIM == 512) sum_reduce<256>(sum_a, sum_b);
    else if (TDIM == 256) sum_reduce<128>(sum_a, sum_b);
    else if (TDIM == 128) sum_reduce<64>(sum_a, sum_b);
    else if (TDIM == 64) sum_reduce<32>(sum_a, sum_b);
    else if (TDIM == 32) sum_reduce<16>(sum_a, sum_b);
    else if (TDIM == 16) sum_reduce<8>(sum_a, sum_b);
    else if (TDIM == 8) sum_reduce<4>(sum_a, sum_b);

    if (TID == 0) {
        result[0] = sum_a[0];
        result[1] = sum_b[0];
    }
}

}  // extern "C"


