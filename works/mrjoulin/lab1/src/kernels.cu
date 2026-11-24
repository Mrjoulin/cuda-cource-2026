#include "kernels.h"
#include <cuda_runtime.h>

#include <algorithm>

namespace lab1 {

namespace {

// Each thread in the noncoalesced variants owns a contiguous chunk of
// chunk_size elements. Adjacent threads therefore access memory with a stride
// of chunk_size floats, breaking warp coalescing.
constexpr std::size_t chunk_size = 64;

}  // namespace

// ---------------------------------------------------------------------------
// Coalesced kernels: consecutive threads read/write consecutive floats.
// ---------------------------------------------------------------------------

__global__ void VectorAddCoalesced(const float* a, const float* b, float* c,
                                   std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

__global__ void VectorSaxpyCoalesced(const float* x, float* y, float alpha,
                                     std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    y[i] = alpha * x[i] + y[i];
  }
}

__global__ void VectorReluCoalesced(const float* x, float* y, std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Noncoalesced kernels: each thread owns a contiguous chunk.
// ---------------------------------------------------------------------------

__global__ void VectorAddNoncoalesced(const float* a, const float* b, float* c,
                                      std::size_t n) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t start = tid * chunk_size;
  const std::size_t end = min(start + chunk_size, n);

  for (std::size_t i = start; i < end; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void VectorSaxpyNoncoalesced(const float* x, float* y, float alpha,
                                        std::size_t n) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t start = tid * chunk_size;
  const std::size_t end = min(start + chunk_size, n);

  for (std::size_t i = start; i < end; ++i) {
    y[i] = alpha * x[i] + y[i];
  }
}

__global__ void VectorReluNoncoalesced(const float* x, float* y,
                                       std::size_t n) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t start = tid * chunk_size;
  const std::size_t end = min(start + chunk_size, n);

  for (std::size_t i = start; i < end; ++i) {
    y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

void LaunchVectorOp(Op op, AccessMode mode, const float* d_a, const float* d_b,
                    float* d_c, float alpha, std::size_t n, int block_size) {
  if (n == 0) {
    return;
  }

  const int block = block_size;
  int grid = 0;

  if (mode == AccessMode::Coalesced) {
    grid = static_cast<int>((n + block - 1) / block);
  } else {
    const std::size_t total_threads = (n + chunk_size - 1) / chunk_size;
    grid = static_cast<int>((total_threads + block - 1) / block);
  }

  // Run kernel
  if (mode == AccessMode::Coalesced) {
    switch (op) {
      case Op::Add:
        VectorAddCoalesced<<<grid, block>>>(d_a, d_b, d_c, n);
        break;
      case Op::Saxpy:
        VectorSaxpyCoalesced<<<grid, block>>>(d_a, d_c, alpha, n);
        break;
      case Op::Relu:
        VectorReluCoalesced<<<grid, block>>>(d_a, d_c, n);
        break;
    }
  } else {
    switch (op) {
      case Op::Add:
        VectorAddNoncoalesced<<<grid, block>>>(d_a, d_b, d_c, n);
        break;
      case Op::Saxpy:
        VectorSaxpyNoncoalesced<<<grid, block>>>(d_a, d_c, alpha, n);
        break;
      case Op::Relu:
        VectorReluNoncoalesced<<<grid, block>>>(d_a, d_c, n);
        break;
    }
  }
}

}  // namespace lab1
