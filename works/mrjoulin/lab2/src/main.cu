/**
 * @file main.cu
 * @brief Lab 2 submission: C = A * B with naive, tiled and cuBLAS implementations.
 *
 * Reads a problem from --input, multiplies on the GPU and writes C to --output. Logs go to
 * stdout, errors to stderr. Exit codes: 0 success, 1 runtime error, 2 bad arguments or input.
 */

#include "cuda_utils.h"
#include "errors.h"
#include "matmul.h"
#include "matrix_io.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace lab2 {

namespace {

constexpr int kDefaultTile = 16;

struct Options {
    std::string inputPath;
    std::string outputPath;
    Mode mode = Mode::Tiled;
    int tile = kDefaultTile;
};

void printUsage(const char *program) {
    std::printf(
        "Usage: %s --input <file> --output <file> [--mode naive|tiled|cublas] [--tile 8|16|32]\n"
        "       %s --help\n"
        "\n"
        "Computes C = A * B in FP32 on the GPU.\n"
        "\n"
        "Arguments:\n"
        "  --input <file>   text file: \"M K N\", then A (M x K) and B (K x N), row by row\n"
        "  --output <file>  text file: \"M N\", then C (M x N), row by row, 9 significant digits\n"
        "  --mode <mode>    naive  - one thread per element, global memory only\n"
        "                   tiled  - shared memory tiles (default)\n"
        "                   cublas - cuBLAS SGEMM, for comparison\n"
        "  --tile <size>    tile edge for tiled mode: 8, 16 (default) or 32\n"
        "\n"
        "M, K and N must be positive. Values must be finite numbers.\n"
        "Exit codes: 0 success, 1 runtime error, 2 bad arguments or input.\n",
        program, program);
}

Mode parseMode(const std::string &name) {
    if (name == "naive") {
        return Mode::Naive;
    }
    if (name == "tiled") {
        return Mode::Tiled;
    }
    if (name == "cublas") {
        return Mode::Cublas;
    }
    throw InputError("unknown mode '" + name + "' (use naive, tiled or cublas)");
}

int parseTile(const std::string &text) {
    char *end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (text.empty() || *end != '\0' || !isSupportedTile(static_cast<int>(value))) {
        throw InputError("unsupported tile '" + text + "' (use 8, 16 or 32)");
    }
    return static_cast<int>(value);
}

Options parseArgs(const std::vector<std::string> &args) {
    Options options;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string &flag = args[i];
        if (flag != "--input" && flag != "--output" && flag != "--mode" && flag != "--tile") {
            throw InputError("unknown argument '" + flag + "'");
        }
        if (i + 1 >= args.size()) {
            throw InputError("missing value for " + flag);
        }
        const std::string &value = args[++i];

        if (flag == "--input") {
            options.inputPath = value;
        } else if (flag == "--output") {
            options.outputPath = value;
        } else if (flag == "--mode") {
            options.mode = parseMode(value);
        } else {
            options.tile = parseTile(value);
        }
    }

    if (options.inputPath.empty() || options.outputPath.empty()) {
        throw InputError("--input and --output are required");
    }
    return options;
}

const char *modeName(Mode mode) {
    switch (mode) {
    case Mode::Naive:
        return "naive";
    case Mode::Tiled:
        return "tiled";
    default:
        return "cublas";
    }
}

// Runs the selected implementation once and returns the kernel time in milliseconds.
float multiply(const Options &options, const Problem &problem, const DeviceBuffer<float> &d_a,
               const DeviceBuffer<float> &d_b, DeviceBuffer<float> &d_c) {
    const CublasHandle cublas;

    Event start;
    Event stop;
    start.record();
    switch (options.mode) {
    case Mode::Naive:
        launchNaive(d_a.data(), d_b.data(), d_c.data(), problem.m, problem.k, problem.n);
        break;
    case Mode::Tiled:
        launchTiled(d_a.data(), d_b.data(), d_c.data(), problem.m, problem.k, problem.n,
                    options.tile, false);
        break;
    case Mode::Cublas:
        sgemmRowMajor(cublas.get(), d_a.data(), d_b.data(), d_c.data(), problem.m, problem.k,
                      problem.n);
        break;
    }
    stop.record();
    return Event::elapsedMs(start, stop);
}

int run(const Options &options) {
    const Problem problem = readProblem(options.inputPath);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw RuntimeFailure("no CUDA device found");
    }

    const DeviceBuffer<float> d_a = toDevice(problem.a);
    const DeviceBuffer<float> d_b = toDevice(problem.b);
    DeviceBuffer<float> d_c(static_cast<std::size_t>(problem.m) * problem.n);

    const float kernelMs = multiply(options, problem, d_a, d_b, d_c);
    const std::vector<float> c = toHost(d_c);
    writeMatrix(options.outputPath, problem.m, problem.n, c);

    std::printf("mode=%s M=%d K=%d N=%d kernel_ms=%.4f output=%s\n", modeName(options.mode),
                problem.m, problem.k, problem.n, static_cast<double>(kernelMs),
                options.outputPath.c_str());
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    for (const std::string &arg : args) {
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    try {
        return run(parseArgs(args));
    } catch (const InputError &error) {
        std::fprintf(stderr, "error: %s\nRun with --help for usage.\n", error.what());
        return 2;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "runtime error: %s\n", error.what());
        return 1;
    }
}

} // namespace lab2

int main(int argc, char **argv) { return lab2::main(argc, argv); }
