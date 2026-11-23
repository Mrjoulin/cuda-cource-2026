/**
 * @file check.cu
 * @brief Correctness suite: every implementation against an FP64 cuBLAS reference.
 *
 * The reference is DGEMM on the same FP32 inputs converted to double. For each element the
 * allowed error is (gamma(2K, FP32) + gamma(K, FP64)) * S, where S = sum_k |A_ik * B_kj| and
 * gamma(q) = q * u / (1 - q * u). The first term bounds the FP32 result, the second the error
 * of the FP64 reference itself. With small integer inputs every product and partial sum is
 * exact in FP32, so there the result must match exactly.
 */

#include "bench.h"
#include "cuda_utils.h"
#include "matmul.h"

#include <cmath>
#include <cstdio>

namespace lab2 {

namespace {

constexpr double kUnitRoundoffFp32 = 0x1p-24;
constexpr double kUnitRoundoffFp64 = 0x1p-53;

double errorFactor(double q, double unitRoundoff) {
    const double qu = q * unitRoundoff;
    return qu / (1.0 - qu);
}

std::vector<double> toDouble(const std::vector<float> &values, bool absolute) {
    std::vector<double> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = static_cast<double>(values[i]);
        converted[i] = absolute ? std::fabs(v) : v;
    }
    return converted;
}

// FP64 product of the inputs and the per-element error bound.
struct Reference {
    std::vector<double> product;
    std::vector<double> bound;
};

Reference computeReference(cublasHandle_t handle, const std::vector<float> &a,
                           const std::vector<float> &b, const Shape &shape) {
    const std::size_t count = static_cast<std::size_t>(shape.m) * shape.n;

    const DeviceBuffer<double> d_a = toDevice(toDouble(a, false));
    const DeviceBuffer<double> d_b = toDevice(toDouble(b, false));
    const DeviceBuffer<double> d_absA = toDevice(toDouble(a, true));
    const DeviceBuffer<double> d_absB = toDevice(toDouble(b, true));
    DeviceBuffer<double> d_product(count);
    DeviceBuffer<double> d_sums(count);

    dgemmRowMajor(handle, d_a.data(), d_b.data(), d_product.data(), shape.m, shape.k, shape.n);
    dgemmRowMajor(handle, d_absA.data(), d_absB.data(), d_sums.data(), shape.m, shape.k, shape.n);

    const double factor = errorFactor(2.0 * shape.k, kUnitRoundoffFp32) +
                          errorFactor(static_cast<double>(shape.k), kUnitRoundoffFp64);

    std::vector<double> bound = toHost(d_sums);
    for (double &s : bound) {
        s *= factor;
    }
    return {toHost(d_product), bound};
}

struct Stats {
    double maxError = 0.0;
    double maxRatio = 0.0; // error / bound, 1.0 means exactly at the limit
    std::size_t violations = 0;
};

Stats compare(const std::vector<float> &got, const Reference &ref, bool exact) {
    Stats stats;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double value = static_cast<double>(got[i]);
        const double error = std::fabs(value - ref.product[i]);
        const bool bad = !std::isfinite(value) || (exact ? error != 0.0 : error > ref.bound[i]);

        if (!std::isfinite(value)) {
            ++stats.violations;
            continue;
        }
        stats.maxError = std::fmax(stats.maxError, error);
        if (ref.bound[i] > 0.0) {
            stats.maxRatio = std::fmax(stats.maxRatio, error / ref.bound[i]);
        }
        if (bad) {
            ++stats.violations;
        }
    }
    return stats;
}

const std::vector<Shape> kShapes = {
    {1, 1, 1},       {2, 3, 2},       {1, 257, 1},        {257, 1, 129}, {16, 16, 16},
    {31, 33, 35},    {32, 32, 32},    {33, 33, 33},       {64, 17, 95},  {127, 65, 257},
    {256, 256, 256}, {300, 500, 200}, {1000, 1000, 1000},
};

int checkShape(cublasHandle_t handle, const std::vector<Variant> &variants, const Shape &shape,
               unsigned seed, bool integers) {
    const std::size_t sizeA = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t sizeB = static_cast<std::size_t>(shape.k) * shape.n;
    const std::size_t sizeC = static_cast<std::size_t>(shape.m) * shape.n;

    const std::vector<float> a =
        integers ? randomIntegerMatrix(sizeA, seed) : randomMatrix(sizeA, seed);
    const std::vector<float> b =
        integers ? randomIntegerMatrix(sizeB, seed + 1) : randomMatrix(sizeB, seed + 1);

    const DeviceBuffer<float> d_a = toDevice(a);
    const DeviceBuffer<float> d_b = toDevice(b);
    DeviceBuffer<float> d_c(sizeC);
    const Reference ref = computeReference(handle, a, b, shape);

    int failures = 0;
    for (const Variant &variant : variants) {
        variant.run(d_a.data(), d_b.data(), d_c.data(), shape);
        CUDA_CHECK(cudaDeviceSynchronize());

        const Stats stats = compare(toHost(d_c), ref, integers);
        const bool ok = stats.violations == 0;
        failures += ok ? 0 : 1;

        std::printf("| %dx%dx%d | %s | %s | %.3e | %.3f | %zu | %s |\n", shape.m, shape.k, shape.n,
                    integers ? "integer" : "random", variant.name.c_str(), stats.maxError,
                    stats.maxRatio, stats.violations, ok ? "ok" : "FAIL");
    }
    return failures;
}

} // namespace

int runCheck() {
    const CublasHandle handle;
    const std::vector<Variant> variants = makeVariants(handle.get());

    std::printf("| M x K x N | data | variant | max error | max error / bound | bad | result |\n");
    std::printf("|---|---|---|---|---|---|---|\n");

    int failures = 0;
    unsigned seed = 2026;
    for (const Shape &shape : kShapes) {
        failures += checkShape(handle.get(), variants, shape, seed, false);
        failures += checkShape(handle.get(), variants, shape, seed, true);
        seed += 10;
    }

    std::printf("\nfailures: %d\n", failures);
    return failures;
}

} // namespace lab2
