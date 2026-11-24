#include "kernels.h"
#include "config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lab1 {
namespace {

constexpr int warmup_iterations = 2;
constexpr int timed_repeats = 20;       // timed repetitions after warmup
constexpr int result_block_size = 256;  // block size used for the result file
constexpr float tolerance = 1e-5f;      // max abs error treated as a mismatch
const int sweep_block_sizes[] = {64, 128, 256, 512, 1024};  // benchmark sweep

// ---------------------------------------------------------------------------
// Error checking: any CUDA failure is a runtime error (exit code 1).
// ---------------------------------------------------------------------------

void CheckCuda(cudaError_t err, const char* call, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", call, file, line,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

void CheckKernelLaunch(const char* file, int line) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch error at %s:%d: %s\n", file, line,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) CheckCuda((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK_KERNEL() CheckKernelLaunch(__FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Input parsing: "N alpha" followed by exactly N pairs "x y".
// ---------------------------------------------------------------------------

/** Strict numeric token reader: the whole token must parse, values must be
 * finite. */
template <typename Number>
bool ReadNumber(std::istream& file, Number& value) {
  std::string token;
  if (!(file >> token)) {
    return false;
  }
  std::string_view text = token;
  if (text.front() == '+') {
    text.remove_prefix(1);
    if (text.empty() || text.front() == '-') {
      return false;
    }
  }
  const char* end = text.data() + text.size();
  const auto result = std::from_chars(text.data(), end, value);
  return result.ec == std::errc{} && result.ptr == end;
}

struct VectorInput {
  float alpha = 0.0f;
  std::vector<float> x;
  std::vector<float> y;
};

/** Read the input file. Any deviation from the format is invalid input (exit
 * code 2). */
VectorInput ReadInput(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::invalid_argument("Cannot open input file: " + path);
  }

  VectorInput input;
  std::size_t n = 0;
  if (!ReadNumber(file, n) || n == 0 || !ReadNumber(file, input.alpha) ||
      !std::isfinite(input.alpha)) {
    throw std::invalid_argument(
        "Expected a positive element count and a finite alpha at the "
        "start of the input file");
  }

  input.x.resize(n);
  input.y.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!ReadNumber(file, input.x[i]) || !ReadNumber(file, input.y[i]) ||
        !std::isfinite(input.x[i]) || !std::isfinite(input.y[i])) {
      throw std::invalid_argument("Expected two finite floats for pair " +
                                  std::to_string(i + 1) + " in " + path);
    }
  }

  std::string extra;
  if (file >> extra) {
    throw std::invalid_argument("Unexpected data after the last pair in " +
                                path);
  }
  return input;
}

// ---------------------------------------------------------------------------
// Host-side references and comparison helpers
// ---------------------------------------------------------------------------

void ComputeReference(Op op, const std::vector<float>& a,
                      const std::vector<float>& b, std::vector<float>& out,
                      float alpha) {
  const std::size_t n = out.size();

  switch (op) {
    case Op::Add:
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
      }
      break;

    case Op::Saxpy:
      // SAXPY overwrites y; b holds the original y.
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = alpha * a[i] + b[i];
      }
      break;

    case Op::Relu:
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
      }
      break;
  }
}

void CompareWithReference(const std::vector<float>& ref,
                          const std::vector<float>& got, float& max_err,
                          std::size_t& mismatches, float tol = tolerance) {
  max_err = 0.0f;
  mismatches = 0;

  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float err = std::fabs(ref[i] - got[i]);

    if (err > max_err) {
      max_err = err;
    }

    if (err > tol) {
      ++mismatches;
    }
  }
}

std::string OpName(Op op) {
  return op == Op::Add ? "add" : (op == Op::Saxpy ? "saxpy" : "relu");
}

std::string ModeName(AccessMode mode) {
  return mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
}

// ---------------------------------------------------------------------------
// Validation of the kernels on small sizes around the block boundary
// ---------------------------------------------------------------------------

struct ValidationResult {
  std::string op_name;
  std::size_t n;
  float max_err;
  std::size_t mismatches;
};

// Uploads x and y, runs one operation on the device and returns the result.
std::vector<float> RunOnDevice(Op op, AccessMode mode,
                               const std::vector<float>& x,
                               const std::vector<float>& y, float alpha,
                               int block_size) {
  const std::size_t n = x.size();
  std::vector<float> result(n, 0.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(d_a, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  if (op == Op::Add || op == Op::Saxpy) {
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(d_b, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  }

  // For SAXPY the launcher treats d_c as y.
  if (op == Op::Saxpy) {
    CUDA_CHECK(
        cudaMemcpy(d_c, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  }

  LaunchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, block_size);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(result.data(), d_c, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_c));
  if (d_b != nullptr) {
    CUDA_CHECK(cudaFree(d_b));
  }
  return result;
}

ValidationResult ValidateOp(Op op, AccessMode mode, std::size_t n, float alpha,
                            int block_size) {
  std::vector<float> x, y;
  x.resize(n);
  y.resize(n);

  // Reproducible pseudo-random values in [-1, 1]; they cover zeros and both
  // signs.
  std::mt19937 gen(2026u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = dist(gen);
    y[i] = dist(gen);
  }

  std::vector<float> ref(n);
  ComputeReference(op, x, y, ref, alpha);

  const std::vector<float> result =
      RunOnDevice(op, mode, x, y, alpha, block_size);

  ValidationResult res;
  res.op_name = OpName(op);
  res.n = n;
  CompareWithReference(ref, result, res.max_err, res.mismatches);

  return res;
}

void RunValidationSuite(AccessMode mode, float alpha) {
  // Sizes around the default block boundary plus tiny and odd sizes.
  const std::size_t sizes[] = {1,   2,   63,   64,   65,  255,
                               256, 257, 1023, 1024, 1025};

  std::cout << "\n### Validation (kernels vs CPU reference)\n\n";
  std::cout << "| Operation | Size | Max abs error | Mismatches (>1e-5) |\n";
  std::cout << "|-----------|------|---------------|--------------------|\n";

  for (std::size_t n : sizes) {
    for (auto op : {Op::Add, Op::Saxpy, Op::Relu}) {
      const ValidationResult r =
          ValidateOp(op, mode, n, alpha, result_block_size);
      std::cout << "| " << std::setw(9) << std::left << r.op_name << " | "
                << std::setw(4) << r.n << " | " << std::scientific
                << std::setprecision(2) << r.max_err << " | " << r.mismatches
                << " |\n";
    }
  }

  std::cout << std::defaultfloat << std::setprecision(6);
}

// ---------------------------------------------------------------------------
// Benchmark: H2D / kernel / D2H / full cycle, median over repeated runs
// ---------------------------------------------------------------------------

struct Timing {
  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  float full_ms = 0.0f;
};

struct BenchmarkResult {
  std::string op_name;
  int block_size;
  Timing median;
  float alloc_ms = 0.0f;
};

float MedianOf(std::vector<float>& values) {
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(middle),
                   values.end());
  return values[middle];
}

struct BenchmarkBuffers {
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
};

// Allocates device memory once and returns the allocation time in ms.
float AllocateBuffers(Op op, std::size_t n, BenchmarkBuffers& buffers) {
  cudaEvent_t alloc_start, alloc_stop;
  CUDA_CHECK(cudaEventCreate(&alloc_start));
  CUDA_CHECK(cudaEventCreate(&alloc_stop));

  CUDA_CHECK(cudaEventRecord(alloc_start));

  CUDA_CHECK(cudaMalloc(&buffers.d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_c, n * sizeof(float)));
  if (op == Op::Add || op == Op::Saxpy) {
    CUDA_CHECK(cudaMalloc(&buffers.d_b, n * sizeof(float)));
  }

  CUDA_CHECK(cudaEventRecord(alloc_stop));
  CUDA_CHECK(cudaEventSynchronize(alloc_stop));
  float alloc_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&alloc_ms, alloc_start, alloc_stop));
  CUDA_CHECK(cudaEventDestroy(alloc_start));
  CUDA_CHECK(cudaEventDestroy(alloc_stop));
  return alloc_ms;
}

void FreeBuffers(const BenchmarkBuffers& buffers) {
  CUDA_CHECK(cudaFree(buffers.d_a));
  CUDA_CHECK(cudaFree(buffers.d_c));
  if (buffers.d_b != nullptr) {
    CUDA_CHECK(cudaFree(buffers.d_b));
  }
}

struct BenchmarkEvents {
  cudaEvent_t h2d_start, h2d_stop, kernel_start, kernel_stop, d2h_start,
      d2h_stop, full_start, full_stop;
};

BenchmarkEvents CreateEvents() {
  BenchmarkEvents ev;
  CUDA_CHECK(cudaEventCreate(&ev.h2d_start));
  CUDA_CHECK(cudaEventCreate(&ev.h2d_stop));
  CUDA_CHECK(cudaEventCreate(&ev.kernel_start));
  CUDA_CHECK(cudaEventCreate(&ev.kernel_stop));
  CUDA_CHECK(cudaEventCreate(&ev.d2h_start));
  CUDA_CHECK(cudaEventCreate(&ev.d2h_stop));
  CUDA_CHECK(cudaEventCreate(&ev.full_start));
  CUDA_CHECK(cudaEventCreate(&ev.full_stop));
  return ev;
}

void DestroyEvents(const BenchmarkEvents& ev) {
  CUDA_CHECK(cudaEventDestroy(ev.h2d_start));
  CUDA_CHECK(cudaEventDestroy(ev.h2d_stop));
  CUDA_CHECK(cudaEventDestroy(ev.kernel_start));
  CUDA_CHECK(cudaEventDestroy(ev.kernel_stop));
  CUDA_CHECK(cudaEventDestroy(ev.d2h_start));
  CUDA_CHECK(cudaEventDestroy(ev.d2h_stop));
  CUDA_CHECK(cudaEventDestroy(ev.full_start));
  CUDA_CHECK(cudaEventDestroy(ev.full_stop));
}

// One timed cycle: H2D copies, kernel, D2H copy.
Timing MeasureIteration(Op op, AccessMode mode, const std::vector<float>& x,
                        const std::vector<float>& y, float alpha,
                        int block_size, const BenchmarkBuffers& buffers,
                        const BenchmarkEvents& ev,
                        std::vector<float>& result_host) {
  const std::size_t n = x.size();

  CUDA_CHECK(cudaEventRecord(ev.full_start));

  // H2D copies: both inputs are re-uploaded every iteration so the
  // measurement includes the real transfer cost.
  CUDA_CHECK(cudaEventRecord(ev.h2d_start));
  CUDA_CHECK(cudaMemcpy(buffers.d_a, x.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  if (op == Op::Add) {
    CUDA_CHECK(cudaMemcpy(buffers.d_b, y.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
  } else if (op == Op::Saxpy) {
    // y is the second operand and the destination.
    CUDA_CHECK(cudaMemcpy(buffers.d_c, y.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaEventRecord(ev.h2d_stop));

  // Kernel.
  CUDA_CHECK(cudaEventRecord(ev.kernel_start));
  LaunchVectorOp(op, mode, buffers.d_a, buffers.d_b, buffers.d_c, alpha, n,
                 block_size);
  CUDA_CHECK(cudaEventRecord(ev.kernel_stop));

  // D2H.
  CUDA_CHECK(cudaEventRecord(ev.d2h_start));
  CUDA_CHECK(cudaMemcpy(result_host.data(), buffers.d_c, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev.d2h_stop));

  CUDA_CHECK(cudaEventRecord(ev.full_stop));
  CUDA_CHECK(cudaEventSynchronize(ev.full_stop));

  Timing t;
  CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, ev.h2d_start, ev.h2d_stop));
  CUDA_CHECK(
      cudaEventElapsedTime(&t.kernel_ms, ev.kernel_start, ev.kernel_stop));
  CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, ev.d2h_start, ev.d2h_stop));
  CUDA_CHECK(cudaEventElapsedTime(&t.full_ms, ev.full_start, ev.full_stop));
  return t;
}

BenchmarkResult BenchmarkOp(Op op, AccessMode mode, const std::vector<float>& x,
                            const std::vector<float>& y, float alpha,
                            int block_size, int repeats) {
  const std::size_t n = x.size();

  BenchmarkResult result;
  result.op_name = OpName(op);
  result.block_size = block_size;

  BenchmarkBuffers buffers;
  result.alloc_ms = AllocateBuffers(op, n, buffers);

  std::vector<float> result_host(n);
  const BenchmarkEvents ev = CreateEvents();

  std::vector<Timing> timings;
  timings.reserve(static_cast<std::size_t>(repeats + warmup_iterations));

  for (int iter = 0; iter < warmup_iterations + repeats; ++iter) {
    const Timing t = MeasureIteration(op, mode, x, y, alpha, block_size,
                                      buffers, ev, result_host);
    if (iter >= warmup_iterations) {
      timings.push_back(t);
    }
  }

  DestroyEvents(ev);
  FreeBuffers(buffers);

  std::vector<float> h2d, kern, d2h, full;
  for (const auto& t : timings) {
    h2d.push_back(t.h2d_ms);
    kern.push_back(t.kernel_ms);
    d2h.push_back(t.d2h_ms);
    full.push_back(t.full_ms);
  }

  result.median.h2d_ms = MedianOf(h2d);
  result.median.kernel_ms = MedianOf(kern);
  result.median.d2h_ms = MedianOf(d2h);
  result.median.full_ms = MedianOf(full);

  return result;
}

void RunBenchmark(AccessMode mode, const VectorInput& input) {
  std::cout << "\n### Benchmark: " << ModeName(mode) << " access\n\n";
  std::cout << "| Operation | Block | H2D, ms | Kernel, ms | D2H, ms | Full "
               "cycle, ms | Alloc, "
            << "ms |\n";
  std::cout << "|-----------|-------|---------|------------|---------|---------"
               "-------|------"
            << "-----|\n";

  for (int block_size : sweep_block_sizes) {
    for (auto op : {Op::Add, Op::Saxpy, Op::Relu}) {
      const BenchmarkResult r = BenchmarkOp(
          op, mode, input.x, input.y, input.alpha, block_size, timed_repeats);
      std::cout << "| " << std::setw(9) << std::left << r.op_name << " | "
                << std::setw(5) << r.block_size << " | " << std::fixed
                << std::setprecision(3) << r.median.h2d_ms << " | "
                << r.median.kernel_ms << " | " << r.median.d2h_ms << " | "
                << r.median.full_ms << " | " << r.alloc_ms << " |\n";
    }
  }

  std::cout << std::defaultfloat << std::setprecision(6);
}

// ---------------------------------------------------------------------------
// Result computation and writing
// ---------------------------------------------------------------------------

struct VectorResults {
  std::vector<float> add;
  std::vector<float> saxpy;
  std::vector<float> relu;
};

/** Run all three operations on the loaded input and copy the results back. */
void ComputeAllResults(AccessMode mode, const VectorInput& input,
                       VectorResults& results, int block_size) {
  const std::size_t n = input.x.size();
  const std::size_t bytes = n * sizeof(float);

  results.add.assign(n, 0.0f);
  results.saxpy.assign(n, 0.0f);
  results.relu.assign(n, 0.0f);

  float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, input.x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, input.y.data(), bytes, cudaMemcpyHostToDevice));

  LaunchVectorOp(Op::Add, mode, d_x, d_y, d_out, input.alpha, n, block_size);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.add.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // SAXPY computes y = alpha * x + y in place, so the destination buffer must
  // start with the original y; d_out currently holds the add results.
  CUDA_CHECK(cudaMemcpy(d_out, input.y.data(), bytes, cudaMemcpyHostToDevice));
  LaunchVectorOp(Op::Saxpy, mode, d_x, nullptr, d_out, input.alpha, n,
                 block_size);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.saxpy.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  LaunchVectorOp(Op::Relu, mode, d_x, nullptr, d_out, input.alpha, n,
                 block_size);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.relu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_out));
}

/** Write "add saxpy relu" lines to the --output file. Failure is a runtime
 * error (exit code 1). */
void WriteResults(const std::string& path, const VectorResults& results) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open output file: " + path);
  }

  file << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (std::size_t i = 0; i < results.add.size(); ++i) {
    file << results.add[i] << ' ' << results.saxpy[i] << ' ' << results.relu[i]
         << '\n';
  }

  file.close();
  if (!file) {
    throw std::runtime_error("Cannot write output file: " + path);
  }
}

}  // namespace
}  // namespace lab1

int main(int argc, char** argv) {
  using namespace lab1;

  try {
    const Config cfg = ParseArgs(argc, argv);
    const VectorInput input = ReadInput(cfg.input_path);
    const std::size_t n = input.x.size();

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << " (compute " << prop.major << "."
              << prop.minor << ")\n";
    std::cout << "Mode: " << ModeName(cfg.mode) << "\n";
    std::cout << "Elements: " << n << ", alpha: " << input.alpha << "\n";

    // Kernel checks on sizes around the block boundary.
    RunValidationSuite(cfg.mode, input.alpha);

    // The actual results: run the loaded input through all three operations.
    VectorResults results;
    ComputeAllResults(cfg.mode, input, results, result_block_size);

    // Cross-check the produced results against CPU references on the same
    // input.
    std::vector<float> ref_add(n), ref_saxpy(n), ref_relu(n);
    ComputeReference(Op::Add, input.x, input.y, ref_add, input.alpha);
    ComputeReference(Op::Saxpy, input.x, input.y, ref_saxpy, input.alpha);
    ComputeReference(Op::Relu, input.x, input.y, ref_relu, input.alpha);

    float max_err = 0.0f;
    std::size_t mismatches = 0;
    CompareWithReference(ref_add, results.add, max_err, mismatches);
    CompareWithReference(ref_saxpy, results.saxpy, max_err, mismatches);
    CompareWithReference(ref_relu, results.relu, max_err, mismatches);

    std::cout << "\nCorrectness on the input data: max abs error = " << max_err
              << ", mismatches = " << mismatches << "\n";

    // Timing measurements, printed to stdout as required by the contract.
    RunBenchmark(cfg.mode, input);

    WriteResults(cfg.output_path, results);
    std::cout << "\nResults written to: " << cfg.output_path << "\n";
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << "\n";
    return exit_usage_error;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return exit_runtime_error;
  }

  return exit_ok;
}
