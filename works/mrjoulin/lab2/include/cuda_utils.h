/**
 * @file cuda_utils.h
 * @brief Error-checking macros, device buffer and event wrappers.
 */

#ifndef LAB2_CUDA_UTILS_H
#define LAB2_CUDA_UTILS_H

#include "errors.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                                 \
            throw lab2::RuntimeFailure(std::string("CUDA error in ") + #call + " at " + __FILE__ + \
                                       ":" + std::to_string(__LINE__) + ": " +                     \
                                       cudaGetErrorString(err_));                                  \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        const cublasStatus_t st_ = (call);                                                         \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                        \
            throw lab2::RuntimeFailure(std::string("cuBLAS error in ") + #call + " at " +          \
                                       __FILE__ + ":" + std::to_string(__LINE__) + ": " +          \
                                       cublasGetStatusString(st_));                                \
        }                                                                                          \
    } while (0)

namespace lab2 {

/** Owns a device allocation. Not copyable, movable. */
template <typename T> class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    T *data() const { return ptr_; }
    std::size_t size() const { return count_; }

  private:
    T *ptr_ = nullptr;
    std::size_t count_ = 0;
};

/** Allocate a device buffer and copy the host vector into it. */
template <typename T> DeviceBuffer<T> toDevice(const std::vector<T> &host) {
    DeviceBuffer<T> device(host.size());
    CUDA_CHECK(
        cudaMemcpy(device.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

/** Copy a device buffer into a new host vector. */
template <typename T> std::vector<T> toHost(const DeviceBuffer<T> &device) {
    std::vector<T> host(device.size());
    CUDA_CHECK(
        cudaMemcpy(host.data(), device.data(), device.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

/** CUDA event for timing on a stream. */
class Event {
  public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }

    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;

    void record(cudaStream_t stream = 0) const { CUDA_CHECK(cudaEventRecord(event_, stream)); }

    /** Milliseconds between two recorded events. Waits for the later one. */
    static float elapsedMs(const Event &start, const Event &stop) {
        CUDA_CHECK(cudaEventSynchronize(stop.event_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

  private:
    cudaEvent_t event_ = nullptr;
};

/** cuBLAS handle in strict FP32 mode (no TF32 or reduced-precision shortcuts). */
class CublasHandle {
  public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
    }
    ~CublasHandle() { cublasDestroy(handle_); }

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;

    cublasHandle_t get() const { return handle_; }

  private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace lab2

#endif // LAB2_CUDA_UTILS_H
