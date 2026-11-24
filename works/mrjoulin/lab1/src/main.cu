/**
 * @file main.cu
 * @brief Lab 1 submission: vector add, SAXPY and ReLU with coalesced and
 *        noncoalesced global-memory access.
 *
 * The program reads vectors from --input ("N alpha" followed by N pairs "x y"),
 * runs the three operations on the GPU in the access mode chosen with --mode,
 * checks them against CPU references and writes "add saxpy relu" lines to
 * --output. Timing measurements are printed to stdout.
 */

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

#include "config.h"
#include "kernels.h"

namespace lab1 {
namespace {

constexpr int kWarmupIterations = 2;
constexpr int kRepeats = 20;           // timed repetitions after warmup
constexpr int kResultBlockSize = 256;  // block size used for the result file
constexpr float kTolerance = 1e-5f;    // max abs error treated as a mismatch
const int kSweepBlockSizes[] = {64, 128, 256, 512, 1024};  // benchmark sweep

// ---------------------------------------------------------------------------
// Error checking: any CUDA failure is a runtime error (exit code 1).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call, __FILE__, \
                   __LINE__, cudaGetErrorString(err));                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                \
  do {                                                                     \
    cudaError_t err = cudaGetLastError();                                  \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "Kernel launch error at %s:%d: %s\n", __FILE__, \
                   __LINE__, cudaGetErrorString(err));                     \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Input parsing: "N alpha" followed by exactly N pairs "x y".
// ---------------------------------------------------------------------------

/** Strict numeric token reader: the whole token must parse, values must be
 * finite. */
template <typename Number>
bool readNumber(std::istream& file, Number& value) {
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
VectorInput readInput(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::invalid_argument("Cannot open input file: " + path);
  }

  VectorInput input;
  std::size_t n = 0;
  if (!readNumber(file, n) || n == 0 || !readNumber(file, input.alpha) ||
      !std::isfinite(input.alpha)) {
    throw std::invalid_argument(
        "Expected a positive element count and a finite alpha at the "
        "start of the input file");
  }

  input.x.resize(n);
  input.y.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!readNumber(file, input.x[i]) || !readNumber(file, input.y[i]) ||
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

void computeReference(Op op, const std::vector<float>& a,
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

void compareWithReference(const std::vector<float>& ref,
                          const std::vector<float>& got, float& maxErr,
                          std::size_t& mismatches, float tol = kTolerance) {
  maxErr = 0.0f;
  mismatches = 0;

  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float err = std::fabs(ref[i] - got[i]);

    if (err > maxErr) {
      maxErr = err;
    }

    if (err > tol) {
      ++mismatches;
    }
  }
}

std::string opName(Op op) {
  return op == Op::Add ? "add" : (op == Op::Saxpy ? "saxpy" : "relu");
}

std::string modeName(AccessMode mode) {
  return mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
}

// ---------------------------------------------------------------------------
// Validation of the kernels on small sizes around the block boundary
// ---------------------------------------------------------------------------

struct ValidationResult {
  std::string opName;
  std::size_t n;
  float maxErr;
  std::size_t mismatches;
};

ValidationResult validateOp(Op op, AccessMode mode, std::size_t n, float alpha,
                            int blockSize) {
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
  computeReference(op, x, y, ref, alpha);

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

  launchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, blockSize);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(result.data(), d_c, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_c));
  if (d_b != nullptr) {
    CUDA_CHECK(cudaFree(d_b));
  }

  ValidationResult res;
  res.opName = opName(op);
  res.n = n;
  compareWithReference(ref, result, res.maxErr, res.mismatches);

  return res;
}

void runValidationSuite(AccessMode mode, float alpha) {
  // Sizes around the default block boundary plus tiny and odd sizes.
  const std::size_t sizes[] = {1,   2,   63,   64,   65,  255,
                               256, 257, 1023, 1024, 1025};

  std::cout << "\n### Validation (kernels vs CPU reference)\n\n";
  std::cout << "| Operation | Size | Max abs error | Mismatches (>1e-5) |\n";
  std::cout << "|-----------|------|---------------|--------------------|\n";

  for (std::size_t n : sizes) {
    for (auto op : {Op::Add, Op::Saxpy, Op::Relu}) {
      const ValidationResult r =
          validateOp(op, mode, n, alpha, kResultBlockSize);
      std::cout << "| " << std::setw(9) << std::left << r.opName << " | "
                << std::setw(4) << r.n << " | " << std::scientific
                << std::setprecision(2) << r.maxErr << " | " << r.mismatches
                << " |\n";
    }
  }

  std::cout << std::defaultfloat << std::setprecision(6);
}

// ---------------------------------------------------------------------------
// Benchmark: H2D / kernel / D2H / full cycle, median over repeated runs
// ---------------------------------------------------------------------------

struct Timing {
  float h2dMs = 0.0f;
  float kernelMs = 0.0f;
  float d2hMs = 0.0f;
  float fullMs = 0.0f;
};

struct BenchmarkResult {
  std::string opName;
  int blockSize;
  Timing median;
  float allocMs = 0.0f;
};

float medianOf(std::vector<float>& values) {
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(middle),
                   values.end());
  return values[middle];
}

BenchmarkResult benchmarkOp(Op op, AccessMode mode, const std::vector<float>& x,
                            const std::vector<float>& y, float alpha,
                            int blockSize, int repeats) {
  const std::size_t n = x.size();

  BenchmarkResult result;
  result.opName = opName(op);
  result.blockSize = blockSize;

  // Allocate device memory once and measure allocation time separately.
  cudaEvent_t allocStart, allocStop;
  CUDA_CHECK(cudaEventCreate(&allocStart));
  CUDA_CHECK(cudaEventCreate(&allocStop));

  CUDA_CHECK(cudaEventRecord(allocStart));

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
  if (op == Op::Add || op == Op::Saxpy) {
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  }

  CUDA_CHECK(cudaEventRecord(allocStop));
  CUDA_CHECK(cudaEventSynchronize(allocStop));
  CUDA_CHECK(cudaEventElapsedTime(&result.allocMs, allocStart, allocStop));
  CUDA_CHECK(cudaEventDestroy(allocStart));
  CUDA_CHECK(cudaEventDestroy(allocStop));

  std::vector<float> resultHost(n);

  cudaEvent_t h2dStart, h2dStop, kernelStart, kernelStop, d2hStart, d2hStop,
      fullStart, fullStop;
  CUDA_CHECK(cudaEventCreate(&h2dStart));
  CUDA_CHECK(cudaEventCreate(&h2dStop));
  CUDA_CHECK(cudaEventCreate(&kernelStart));
  CUDA_CHECK(cudaEventCreate(&kernelStop));
  CUDA_CHECK(cudaEventCreate(&d2hStart));
  CUDA_CHECK(cudaEventCreate(&d2hStop));
  CUDA_CHECK(cudaEventCreate(&fullStart));
  CUDA_CHECK(cudaEventCreate(&fullStop));

  std::vector<Timing> timings;
  timings.reserve(static_cast<std::size_t>(repeats + kWarmupIterations));

  for (int iter = 0; iter < kWarmupIterations + repeats; ++iter) {
    CUDA_CHECK(cudaEventRecord(fullStart));

    // H2D copies: both inputs are re-uploaded every iteration so the
    // measurement includes the real transfer cost.
    CUDA_CHECK(cudaEventRecord(h2dStart));
    CUDA_CHECK(
        cudaMemcpy(d_a, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    if (op == Op::Add) {
      CUDA_CHECK(
          cudaMemcpy(d_b, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    } else if (op == Op::Saxpy) {
      // y is the second operand and the destination.
      CUDA_CHECK(
          cudaMemcpy(d_c, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaEventRecord(h2dStop));

    // Kernel.
    CUDA_CHECK(cudaEventRecord(kernelStart));
    launchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, blockSize);
    CUDA_CHECK(cudaEventRecord(kernelStop));

    // D2H.
    CUDA_CHECK(cudaEventRecord(d2hStart));
    CUDA_CHECK(cudaMemcpy(resultHost.data(), d_c, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2hStop));

    CUDA_CHECK(cudaEventRecord(fullStop));
    CUDA_CHECK(cudaEventSynchronize(fullStop));

    if (iter >= kWarmupIterations) {
      Timing t;
      CUDA_CHECK(cudaEventElapsedTime(&t.h2dMs, h2dStart, h2dStop));
      CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, kernelStart, kernelStop));
      CUDA_CHECK(cudaEventElapsedTime(&t.d2hMs, d2hStart, d2hStop));
      CUDA_CHECK(cudaEventElapsedTime(&t.fullMs, fullStart, fullStop));
      timings.push_back(t);
    }
  }

  CUDA_CHECK(cudaEventDestroy(h2dStart));
  CUDA_CHECK(cudaEventDestroy(h2dStop));
  CUDA_CHECK(cudaEventDestroy(kernelStart));
  CUDA_CHECK(cudaEventDestroy(kernelStop));
  CUDA_CHECK(cudaEventDestroy(d2hStart));
  CUDA_CHECK(cudaEventDestroy(d2hStop));
  CUDA_CHECK(cudaEventDestroy(fullStart));
  CUDA_CHECK(cudaEventDestroy(fullStop));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_c));
  if (d_b != nullptr) {
    CUDA_CHECK(cudaFree(d_b));
  }

  std::vector<float> h2d, kern, d2h, full;
  for (const auto& t : timings) {
    h2d.push_back(t.h2dMs);
    kern.push_back(t.kernelMs);
    d2h.push_back(t.d2hMs);
    full.push_back(t.fullMs);
  }

  result.median.h2dMs = medianOf(h2d);
  result.median.kernelMs = medianOf(kern);
  result.median.d2hMs = medianOf(d2h);
  result.median.fullMs = medianOf(full);

  return result;
}

void runBenchmark(AccessMode mode, const VectorInput& input) {
  std::cout << "\n### Benchmark: " << modeName(mode) << " access\n\n";
  std::cout << "| Operation | Block | H2D, ms | Kernel, ms | D2H, ms | Full "
               "cycle, ms | Alloc, "
            << "ms |\n";
  std::cout << "|-----------|-------|---------|------------|---------|---------"
               "-------|------"
            << "-----|\n";

  for (int blockSize : kSweepBlockSizes) {
    for (auto op : {Op::Add, Op::Saxpy, Op::Relu}) {
      const BenchmarkResult r = benchmarkOp(op, mode, input.x, input.y,
                                            input.alpha, blockSize, kRepeats);
      std::cout << "| " << std::setw(9) << std::left << r.opName << " | "
                << std::setw(5) << r.blockSize << " | " << std::fixed
                << std::setprecision(3) << r.median.h2dMs << " | "
                << r.median.kernelMs << " | " << r.median.d2hMs << " | "
                << r.median.fullMs << " | " << r.allocMs << " |\n";
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
void computeAllResults(AccessMode mode, const VectorInput& input,
                       VectorResults& results, int blockSize) {
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

  launchVectorOp(Op::Add, mode, d_x, d_y, d_out, input.alpha, n, blockSize);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.add.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // SAXPY computes y = alpha * x + y in place, so the destination buffer must
  // start with the original y; d_out currently holds the add results.
  CUDA_CHECK(cudaMemcpy(d_out, input.y.data(), bytes, cudaMemcpyHostToDevice));
  launchVectorOp(Op::Saxpy, mode, d_x, nullptr, d_out, input.alpha, n,
                 blockSize);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.saxpy.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  launchVectorOp(Op::Relu, mode, d_x, nullptr, d_out, input.alpha, n,
                 blockSize);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(
      cudaMemcpy(results.relu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_out));
}

/** Write "add saxpy relu" lines to the --output file. Failure is a runtime
 * error (exit code 1). */
void writeResults(const std::string& path, const VectorResults& results) {
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
    const Config cfg = parseArgs(argc, argv);
    const VectorInput input = readInput(cfg.inputPath);
    const std::size_t n = input.x.size();

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << " (compute " << prop.major << "."
              << prop.minor << ")\n";
    std::cout << "Mode: " << modeName(cfg.mode) << "\n";
    std::cout << "Elements: " << n << ", alpha: " << input.alpha << "\n";

    // Kernel checks on sizes around the block boundary.
    runValidationSuite(cfg.mode, input.alpha);

    // The actual results: run the loaded input through all three operations.
    VectorResults results;
    computeAllResults(cfg.mode, input, results, kResultBlockSize);

    // Cross-check the produced results against CPU references on the same
    // input.
    std::vector<float> refAdd(n), refSaxpy(n), refRelu(n);
    computeReference(Op::Add, input.x, input.y, refAdd, input.alpha);
    computeReference(Op::Saxpy, input.x, input.y, refSaxpy, input.alpha);
    computeReference(Op::Relu, input.x, input.y, refRelu, input.alpha);

    float maxErr = 0.0f;
    std::size_t mismatches = 0;
    compareWithReference(refAdd, results.add, maxErr, mismatches);
    compareWithReference(refSaxpy, results.saxpy, maxErr, mismatches);
    compareWithReference(refRelu, results.relu, maxErr, mismatches);

    std::cout << "\nCorrectness on the input data: max abs error = " << maxErr
              << ", mismatches = " << mismatches << "\n";

    // Timing measurements, printed to stdout as required by the contract.
    runBenchmark(cfg.mode, input);

    writeResults(cfg.outputPath, results);
    std::cout << "\nResults written to: " << cfg.outputPath << "\n";
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << "\n";
    return kExitUsageError;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return kExitRuntimeError;
  }

  return kExitOk;
}
