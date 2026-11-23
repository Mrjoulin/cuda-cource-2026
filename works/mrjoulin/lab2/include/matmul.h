/**
 * @file matmul.h
 * @brief Matrix multiplication C = A * B: naive and tiled kernels, cuBLAS reference, bank
 *        conflict test.
 *
 * All matrices are row-major float (double for the FP64 reference). A is m x k, B is k x n,
 * C is m x n. Pointers are device pointers.
 */

#ifndef LAB2_MATMUL_H
#define LAB2_MATMUL_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace lab2 {

/** Implementation selected with --mode. */
enum class Mode { Naive, Tiled, Cublas };

/** Block edge of the naive kernel (16 x 16 threads). */
constexpr int kNaiveBlock = 16;

/** Tile sizes the tiled kernel is instantiated for. */
bool isSupportedTile(int tile);

/** Naive kernel: one thread per C element, every operand comes from global memory. */
void launchNaive(const float *d_a, const float *d_b, float *d_c, int m, int k, int n,
                 cudaStream_t stream = 0);

/**
 * Tiled kernel: a tile x tile block of threads stages tiles of A and B in shared memory.
 * With pad = true the shared arrays get one extra column (tile x (tile + 1)).
 */
void launchTiled(const float *d_a, const float *d_b, float *d_c, int m, int k, int n, int tile,
                 bool pad, cudaStream_t stream = 0);

/** cuBLAS SGEMM on row-major data, strict FP32. */
void sgemmRowMajor(cublasHandle_t handle, const float *d_a, const float *d_b, float *d_c, int m,
                   int k, int n);

/** cuBLAS DGEMM on row-major data, used as the FP64 reference. */
void dgemmRowMajor(cublasHandle_t handle, const double *d_a, const double *d_b, double *d_c, int m,
                   int k, int n);

/** Static resources of a compiled kernel. */
struct KernelInfo {
    int registersPerThread;
    std::size_t sharedBytesPerBlock;
};

KernelInfo naiveInfo();
KernelInfo tiledInfo(int tile, bool pad);

/**
 * Shared memory bank test. Each thread of a warp reads buf[lane * stride + c] in a loop, so the
 * stride decides how many lanes hit the same bank. d_out needs blocks * threads floats.
 */
void launchBankTest(float *d_out, int stride, int iterations, int blocks, int threads,
                    cudaStream_t stream = 0);

} // namespace lab2

#endif // LAB2_MATMUL_H
