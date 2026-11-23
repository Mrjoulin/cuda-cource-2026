/**
 * @file kernels.cu
 * @brief Naive and tiled matmul kernels and the shared memory bank conflict test.
 */

#include "cuda_utils.h"
#include "matmul.h"

namespace lab2 {

namespace {

// CUDA allows at most 65535 blocks along grid.y.
constexpr long long kMaxGridY = 65535;

// Bank test buffer: 32 rows of 33 floats, large enough for strides 1..33.
constexpr int kBankBufferSize = 32 * 33;

int ceilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }

// ---------------------------------------------------------------------------
// Naive kernel
// ---------------------------------------------------------------------------

// Thread (row, col) walks one row of A and one column of B. Threads of a warp share the row and
// have consecutive columns, so B reads coalesce and A reads are a broadcast.
__global__ void matmulNaive(const float *a, const float *b, float *c, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += a[static_cast<std::size_t>(row) * k + i] * b[static_cast<std::size_t>(i) * n + col];
    }
    c[static_cast<std::size_t>(row) * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Tiled kernel
// ---------------------------------------------------------------------------

// A block computes a Tile x Tile piece of C. For every step along K it loads one tile of A and
// one tile of B into shared memory, so each global element is read once per block instead of
// once per thread. Tiles that stick out of the matrix are filled with zeros, which keeps the
// inner product correct for any m, k and n.
template <int Tile, int Pad>
__global__ void matmulTiled(const float *a, const float *b, float *c, int m, int k, int n) {
    __shared__ float tileA[Tile][Tile + Pad];
    __shared__ float tileB[Tile][Tile + Pad];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * Tile + ty;
    const int col = blockIdx.x * Tile + tx;

    float acc = 0.0f;
    const int numTiles = (k + Tile - 1) / Tile;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * Tile + tx;
        const int bRow = t * Tile + ty;

        tileA[ty][tx] = (row < m && aCol < k) ? a[static_cast<std::size_t>(row) * k + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < k && col < n) ? b[static_cast<std::size_t>(bRow) * n + col] : 0.0f;

        // Both tiles must be complete before anyone reads them.
        __syncthreads();

        for (int i = 0; i < Tile; ++i) {
            acc += tileA[ty][i] * tileB[i][tx];
        }

        // Nobody may overwrite the tiles while a slower thread still reads them.
        __syncthreads();
    }

    // Threads outside C still took part in the loads and barriers above. Only the store is
    // guarded.
    if (row < m && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = acc;
    }
}

template <int Tile, int Pad>
void launchTiledImpl(const float *d_a, const float *d_b, float *d_c, int m, int k, int n,
                     cudaStream_t stream) {
    const dim3 block(Tile, Tile);
    const dim3 grid(ceilDiv(n, Tile), ceilDiv(m, Tile));
    matmulTiled<Tile, Pad><<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, k, n);
    CUDA_CHECK(cudaGetLastError());
}

template <int Tile, int Pad> KernelInfo tiledInfoImpl() {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmulTiled<Tile, Pad>));
    return {attr.numRegs, attr.sharedSizeBytes};
}

// ---------------------------------------------------------------------------
// Bank conflict test
// ---------------------------------------------------------------------------

// Lane l reads word l * stride + c. Its bank is (l * stride + c) mod 32, so the number of lanes
// that share a bank is gcd(stride, 32). Stride 33 breaks the pattern and has no conflicts.
// The pointer is volatile so the compiler cannot hoist the loads out of the loop.
__global__ void bankTest(float *out, int stride, int iterations) {
    __shared__ float buf[kBankBufferSize];

    for (int i = threadIdx.x; i < kBankBufferSize; i += blockDim.x) {
        buf[i] = static_cast<float>(i);
    }
    __syncthreads();

    volatile float *vbuf = buf;
    const int lane = threadIdx.x & 31;

    float acc = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        acc += vbuf[lane * stride + (it & 31)];
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

} // namespace

bool isSupportedTile(int tile) { return tile == 8 || tile == 16 || tile == 32; }

void launchNaive(const float *d_a, const float *d_b, float *d_c, int m, int k, int n,
                 cudaStream_t stream) {
    const dim3 block(kNaiveBlock, kNaiveBlock);
    const dim3 grid(ceilDiv(n, kNaiveBlock), ceilDiv(m, kNaiveBlock));
    matmulNaive<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, k, n);
    CUDA_CHECK(cudaGetLastError());
}

void launchTiled(const float *d_a, const float *d_b, float *d_c, int m, int k, int n, int tile,
                 bool pad, cudaStream_t stream) {
    if (!isSupportedTile(tile)) {
        throw InputError("unsupported tile size " + std::to_string(tile) + " (use 8, 16 or 32)");
    }
    if (ceilDiv(m, tile) > kMaxGridY) {
        throw InputError("too many rows for this tile size: grid.y would exceed 65535");
    }

    switch (tile) {
    case 8:
        pad ? launchTiledImpl<8, 1>(d_a, d_b, d_c, m, k, n, stream)
            : launchTiledImpl<8, 0>(d_a, d_b, d_c, m, k, n, stream);
        break;
    case 16:
        pad ? launchTiledImpl<16, 1>(d_a, d_b, d_c, m, k, n, stream)
            : launchTiledImpl<16, 0>(d_a, d_b, d_c, m, k, n, stream);
        break;
    default:
        pad ? launchTiledImpl<32, 1>(d_a, d_b, d_c, m, k, n, stream)
            : launchTiledImpl<32, 0>(d_a, d_b, d_c, m, k, n, stream);
        break;
    }
}

KernelInfo naiveInfo() {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmulNaive));
    return {attr.numRegs, attr.sharedSizeBytes};
}

KernelInfo tiledInfo(int tile, bool pad) {
    switch (tile) {
    case 8:
        return pad ? tiledInfoImpl<8, 1>() : tiledInfoImpl<8, 0>();
    case 16:
        return pad ? tiledInfoImpl<16, 1>() : tiledInfoImpl<16, 0>();
    case 32:
        return pad ? tiledInfoImpl<32, 1>() : tiledInfoImpl<32, 0>();
    default:
        throw InputError("unsupported tile size " + std::to_string(tile));
    }
}

void launchBankTest(float *d_out, int stride, int iterations, int blocks, int threads,
                    cudaStream_t stream) {
    bankTest<<<blocks, threads, 0, stream>>>(d_out, stride, iterations);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lab2
