/**
 * @file bench.h
 * @brief Shared pieces of the bench tool: implementation list, random inputs, correctness suite.
 */

#ifndef LAB2_BENCH_H
#define LAB2_BENCH_H

#include <cublas_v2.h>

#include <functional>
#include <string>
#include <vector>

namespace lab2 {

/** Matrix dimensions of one multiplication. */
struct Shape {
    int m;
    int k;
    int n;
};

/** One implementation under test. */
struct Variant {
    std::string name;
    std::function<void(const float *, const float *, float *, const Shape &)> run;
};

/** naive, tiled 8/16/32, tiled 32 with padding, cuBLAS. The handle must outlive the result. */
std::vector<Variant> makeVariants(cublasHandle_t handle);

/** Uniform random floats in [-1, 1], deterministic for a given seed. */
std::vector<float> randomMatrix(std::size_t count, unsigned seed);

/** Random small integers in [-3, 3]. Products and sums stay exact in FP32. */
std::vector<float> randomIntegerMatrix(std::size_t count, unsigned seed);

/** Median of the values. */
float median(std::vector<float> values);

/** Correctness suite against the FP64 cuBLAS reference. Returns the number of failures. */
int runCheck();

} // namespace lab2

#endif // LAB2_BENCH_H
