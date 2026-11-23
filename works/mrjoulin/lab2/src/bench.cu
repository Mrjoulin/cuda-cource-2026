/**
 * @file bench.cu
 * @brief Experiments for the report. Not part of the graded CLI.
 *
 * Usage: bench <command> [--shapes list] [--repeats N]
 *   check     correctness against the FP64 cuBLAS reference
 *   info      registers and shared memory per kernel, device properties
 *   time      kernel time and GFLOP/s of every variant
 *   transfer  allocation, H2D, kernel and D2H time for one tiled run
 *   banks     shared memory bank conflict test
 * A shape is "N" (square) or "MxKxN". The list is comma separated.
 */

#include "bench.h"
#include "cuda_utils.h"
#include "errors.h"
#include "matmul.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <sstream>
#include <string>

namespace lab2 {

namespace {

constexpr int kWarmup = 3;
constexpr int kDefaultRepeats = 10;
constexpr unsigned kSeed = 2026;

struct BenchOptions {
    std::string command;
    std::vector<Shape> shapes = {{512, 512, 512},
                                 {1024, 1024, 1024},
                                 {2048, 2048, 2048},
                                 {4096, 4096, 4096},
                                 {1000, 1500, 700}};
    int repeats = kDefaultRepeats;
};

Shape parseShape(const std::string &text) {
    int m = 0;
    int k = 0;
    int n = 0;
    if (std::sscanf(text.c_str(), "%dx%dx%d", &m, &k, &n) == 3 && m > 0 && k > 0 && n > 0) {
        return {m, k, n};
    }
    const int side = std::atoi(text.c_str());
    if (side > 0) {
        return {side, side, side};
    }
    throw InputError("bad shape '" + text + "'");
}

std::vector<Shape> parseShapes(const std::string &text) {
    std::vector<Shape> shapes;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        shapes.push_back(parseShape(item));
    }
    if (shapes.empty()) {
        throw InputError("empty shape list");
    }
    return shapes;
}

BenchOptions parseArgs(int argc, char **argv) {
    if (argc < 2) {
        throw InputError("usage: bench check|info|time|transfer|banks [--shapes list] "
                         "[--repeats N]");
    }
    BenchOptions options;
    options.command = argv[1];

    for (int i = 2; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) {
            throw InputError("missing value for " + flag);
        }
        if (flag == "--shapes") {
            options.shapes = parseShapes(argv[i + 1]);
        } else if (flag == "--repeats") {
            options.repeats = std::atoi(argv[i + 1]);
            if (options.repeats <= 0) {
                throw InputError("--repeats must be positive");
            }
        } else {
            throw InputError("unknown argument '" + flag + "'");
        }
    }
    return options;
}

double gflops(const Shape &s, float ms) {
    return 2.0 * s.m * s.k * s.n / (static_cast<double>(ms) * 1e6);
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

void printKernelInfo(const char *name, const KernelInfo &info) {
    std::printf("| %s | %d | %zu |\n", name, info.registersPerThread, info.sharedBytesPerBlock);
}

int runInfo() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s, cc %d.%d, %d SMs, %zu KiB shared per block, %d regs per block\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.sharedMemPerBlock / 1024, prop.regsPerBlock);

    std::printf("\n| kernel | registers per thread | static shared memory, bytes |\n");
    std::printf("|---|---|---|\n");
    printKernelInfo("naive16", naiveInfo());
    for (const int tile : {8, 16, 32}) {
        printKernelInfo(("tiled" + std::to_string(tile)).c_str(), tiledInfo(tile, false));
    }
    printKernelInfo("tiled32pad", tiledInfo(32, true));
    return 0;
}

// ---------------------------------------------------------------------------
// time
// ---------------------------------------------------------------------------

// Times only the kernel: data is already on the device and the output buffer is reused.
std::vector<float> timeVariant(const Variant &variant, const DeviceBuffer<float> &d_a,
                               const DeviceBuffer<float> &d_b, DeviceBuffer<float> &d_c,
                               const Shape &shape, int repeats) {
    for (int i = 0; i < kWarmup; ++i) {
        variant.run(d_a.data(), d_b.data(), d_c.data(), shape);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    for (int i = 0; i < repeats; ++i) {
        Event start;
        Event stop;
        start.record();
        variant.run(d_a.data(), d_b.data(), d_c.data(), shape);
        stop.record();
        samples.push_back(Event::elapsedMs(start, stop));
    }
    return samples;
}

int runTime(const BenchOptions &options) {
    const CublasHandle handle;
    const std::vector<Variant> variants = makeVariants(handle.get());

    std::printf("repeats=%d warmup=%d (kernel only, data already on the device)\n\n",
                options.repeats, kWarmup);
    std::printf("| M x K x N | variant | median, ms | min, ms | max, ms | GFLOP/s |\n");
    std::printf("|---|---|---|---|---|---|\n");

    for (const Shape &shape : options.shapes) {
        const DeviceBuffer<float> d_a =
            toDevice(randomMatrix(static_cast<std::size_t>(shape.m) * shape.k, kSeed));
        const DeviceBuffer<float> d_b =
            toDevice(randomMatrix(static_cast<std::size_t>(shape.k) * shape.n, kSeed + 1));
        DeviceBuffer<float> d_c(static_cast<std::size_t>(shape.m) * shape.n);

        for (const Variant &variant : variants) {
            const std::vector<float> samples =
                timeVariant(variant, d_a, d_b, d_c, shape, options.repeats);
            const float med = median(samples);
            const auto [lo, hi] = std::minmax_element(samples.begin(), samples.end());
            std::printf("| %dx%dx%d | %s | %.3f | %.3f | %.3f | %.1f |\n", shape.m, shape.k,
                        shape.n, variant.name.c_str(), static_cast<double>(med),
                        static_cast<double>(*lo), static_cast<double>(*hi), gflops(shape, med));
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// transfer
// ---------------------------------------------------------------------------

double cpuMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from)
        .count();
}

// One full cycle with the tiled kernel (tile 16): allocate, H2D, kernel, D2H. Allocation uses
// the CPU clock because cudaMalloc is a host call. The other steps use CUDA events.
int runTransfer(const BenchOptions &options) {
    CUDA_CHECK(cudaFree(nullptr)); // create the context before timing anything

    std::printf("repeats=%d, tiled16, FP32 (median)\n\n", options.repeats);
    std::printf("| M x K x N | alloc, ms | H2D, ms | kernel, ms | D2H, ms | full cycle, ms |\n");
    std::printf("|---|---|---|---|---|---|\n");

    for (const Shape &shape : options.shapes) {
        const std::vector<float> a =
            randomMatrix(static_cast<std::size_t>(shape.m) * shape.k, kSeed);
        const std::vector<float> b =
            randomMatrix(static_cast<std::size_t>(shape.k) * shape.n, kSeed + 1);
        std::vector<float> c(static_cast<std::size_t>(shape.m) * shape.n);

        std::vector<float> alloc;
        std::vector<float> h2d;
        std::vector<float> kernel;
        std::vector<float> d2h;

        for (int i = 0; i < kWarmup + options.repeats; ++i) {
            const auto allocStart = std::chrono::steady_clock::now();
            DeviceBuffer<float> d_a(a.size());
            DeviceBuffer<float> d_b(b.size());
            DeviceBuffer<float> d_c(c.size());
            CUDA_CHECK(cudaDeviceSynchronize());
            const double allocMs = cpuMs(allocStart);

            Event e0, e1, e2, e3;
            e0.record();
            CUDA_CHECK(
                cudaMemcpy(d_a.data(), a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(
                cudaMemcpy(d_b.data(), b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
            e1.record();
            launchTiled(d_a.data(), d_b.data(), d_c.data(), shape.m, shape.k, shape.n, 16, false);
            e2.record();
            CUDA_CHECK(
                cudaMemcpy(c.data(), d_c.data(), c.size() * sizeof(float), cudaMemcpyDeviceToHost));
            e3.record();

            if (i >= kWarmup) {
                alloc.push_back(static_cast<float>(allocMs));
                h2d.push_back(Event::elapsedMs(e0, e1));
                kernel.push_back(Event::elapsedMs(e1, e2));
                d2h.push_back(Event::elapsedMs(e2, e3));
            }
        }

        const float a_ms = median(alloc);
        const float h_ms = median(h2d);
        const float k_ms = median(kernel);
        const float d_ms = median(d2h);
        std::printf("| %dx%dx%d | %.3f | %.3f | %.3f | %.3f | %.3f |\n", shape.m, shape.k, shape.n,
                    static_cast<double>(a_ms), static_cast<double>(h_ms), static_cast<double>(k_ms),
                    static_cast<double>(d_ms), static_cast<double>(a_ms + h_ms + k_ms + d_ms));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// banks
// ---------------------------------------------------------------------------

int runBanks(const BenchOptions &options) {
    constexpr int kBlocks = 256;
    constexpr int kThreads = 256;
    constexpr int kIterations = 1 << 14;
    constexpr int kBanks = 32;

    DeviceBuffer<float> d_out(static_cast<std::size_t>(kBlocks) * kThreads);

    std::printf("repeats=%d, %d blocks x %d threads, %d loads per thread\n\n", options.repeats,
                kBlocks, kThreads, kIterations);
    std::printf("| stride | lanes per bank | median, ms | vs stride 1 |\n");
    std::printf("|---|---|---|---|\n");

    float baseline = 0.0f;
    for (const int stride : {1, 2, 4, 8, 16, 32, 33}) {
        for (int i = 0; i < kWarmup; ++i) {
            launchBankTest(d_out.data(), stride, kIterations, kBlocks, kThreads);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> samples;
        for (int i = 0; i < options.repeats; ++i) {
            Event start;
            Event stop;
            start.record();
            launchBankTest(d_out.data(), stride, kIterations, kBlocks, kThreads);
            stop.record();
            samples.push_back(Event::elapsedMs(start, stop));
        }

        const float med = median(samples);
        baseline = stride == 1 ? med : baseline;
        std::printf("| %d | %d | %.3f | %.2fx |\n", stride, std::gcd(stride, kBanks),
                    static_cast<double>(med), static_cast<double>(med / baseline));
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const BenchOptions options = parseArgs(argc, argv);

        if (options.command == "check") {
            return runCheck() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (options.command == "info") {
            return runInfo();
        }
        if (options.command == "time") {
            return runTime(options);
        }
        if (options.command == "transfer") {
            return runTransfer(options);
        }
        if (options.command == "banks") {
            return runBanks(options);
        }
        throw InputError("unknown command '" + options.command + "'");
    } catch (const InputError &error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "runtime error: %s\n", error.what());
        return 1;
    }
}

} // namespace lab2

int main(int argc, char **argv) { return lab2::main(argc, argv); }
