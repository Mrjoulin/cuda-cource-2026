/**
 * @file reference.cu
 * @brief cuBLAS wrappers for row-major matrices.
 */

#include "cuda_utils.h"
#include "matmul.h"

namespace lab2 {

// cuBLAS is column-major. A row-major matrix read as column-major is its transpose, so
// C = A * B in row-major is the same memory as C^T = B^T * A^T in column-major. We pass B first,
// then A, and swap m and n.

void sgemmRowMajor(cublasHandle_t handle, const float *d_a, const float *d_b, float *d_c, int m,
                   int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n, d_a, k,
                             &beta, d_c, n));
}

void dgemmRowMajor(cublasHandle_t handle, const double *d_a, const double *d_b, double *d_c, int m,
                   int k, int n) {
    const double alpha = 1.0;
    const double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n, d_a, k,
                             &beta, d_c, n));
}

} // namespace lab2
