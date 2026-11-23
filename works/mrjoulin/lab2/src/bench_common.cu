/**
 * @file bench_common.cu
 * @brief Variant list, random inputs and median helper for the bench tool.
 */

#include "bench.h"
#include "matmul.h"

#include <algorithm>
#include <random>

namespace lab2 {

std::vector<Variant> makeVariants(cublasHandle_t handle) {
    std::vector<Variant> variants;

    variants.push_back({"naive16", [](const float *a, const float *b, float *c, const Shape &s) {
                            launchNaive(a, b, c, s.m, s.k, s.n);
                        }});

    for (const int tile : {8, 16, 32}) {
        variants.push_back({"tiled" + std::to_string(tile),
                            [tile](const float *a, const float *b, float *c, const Shape &s) {
                                launchTiled(a, b, c, s.m, s.k, s.n, tile, false);
                            }});
    }

    variants.push_back({"tiled32pad", [](const float *a, const float *b, float *c, const Shape &s) {
                            launchTiled(a, b, c, s.m, s.k, s.n, 32, true);
                        }});

    variants.push_back(
        {"cublas", [handle](const float *a, const float *b, float *c, const Shape &s) {
             sgemmRowMajor(handle, a, b, c, s.m, s.k, s.n);
         }});

    return variants;
}

std::vector<float> randomMatrix(std::size_t count, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> values(count);
    for (float &v : values) {
        v = dist(gen);
    }
    return values;
}

std::vector<float> randomIntegerMatrix(std::size_t count, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-3, 3);

    std::vector<float> values(count);
    for (float &v : values) {
        v = static_cast<float>(dist(gen));
    }
    return values;
}

float median(std::vector<float> values) {
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

} // namespace lab2
