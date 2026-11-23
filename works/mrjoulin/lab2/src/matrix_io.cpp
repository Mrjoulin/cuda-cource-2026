/**
 * @file matrix_io.cpp
 * @brief Reading and writing matrices in the course text format.
 */

#include "matrix_io.h"

#include "errors.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace lab2 {

namespace {

// Largest matrix we accept, in elements. Keeps every index below 2^31.
constexpr long long kMaxElements = 1LL << 30;

std::string readWholeFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw InputError("cannot open input file: " + path);
    }
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

// Walks over whitespace-separated numbers in a buffer.
class Tokenizer {
  public:
    explicit Tokenizer(const std::string &text) : cursor_(text.c_str()) {}

    int nextDimension(const char *name) {
        skipSpace();
        errno = 0;
        char *end = nullptr;
        const long long value = std::strtoll(cursor_, &end, 10);
        if (end == cursor_ || errno == ERANGE || !atTokenEnd(end)) {
            throw InputError(std::string("expected an integer for ") + name);
        }
        if (value <= 0 || value > kMaxElements) {
            throw InputError(std::string(name) + " must be a positive integer, got " +
                             std::to_string(value));
        }
        cursor_ = end;
        return static_cast<int>(value);
    }

    float nextFloat(std::size_t index, const char *matrix) {
        skipSpace();
        char *end = nullptr;
        const float value = std::strtof(cursor_, &end);
        if (end == cursor_ || !atTokenEnd(end)) {
            throw InputError(std::string("matrix ") + matrix + " has too few or invalid values " +
                             "(failed at element " + std::to_string(index) + ")");
        }
        if (!std::isfinite(value)) {
            throw InputError(std::string("matrix ") + matrix + " contains NaN or infinity");
        }
        cursor_ = end;
        return value;
    }

    bool hasMore() {
        skipSpace();
        return *cursor_ != '\0';
    }

  private:
    void skipSpace() {
        while (*cursor_ == ' ' || *cursor_ == '\t' || *cursor_ == '\n' || *cursor_ == '\r') {
            ++cursor_;
        }
    }

    static bool atTokenEnd(const char *p) {
        return *p == '\0' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r';
    }

    const char *cursor_;
};

std::vector<float> readMatrix(Tokenizer &tokens, std::size_t count, const char *name) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = tokens.nextFloat(i, name);
    }
    return values;
}

} // namespace

Problem readProblem(const std::string &path) {
    const std::string text = readWholeFile(path);
    Tokenizer tokens(text);

    Problem problem;
    problem.m = tokens.nextDimension("M");
    problem.k = tokens.nextDimension("K");
    problem.n = tokens.nextDimension("N");

    const long long sizeA = static_cast<long long>(problem.m) * problem.k;
    const long long sizeB = static_cast<long long>(problem.k) * problem.n;
    const long long sizeC = static_cast<long long>(problem.m) * problem.n;
    if (sizeA > kMaxElements || sizeB > kMaxElements || sizeC > kMaxElements) {
        throw InputError("matrix is too large (more than 2^30 elements)");
    }

    problem.a = readMatrix(tokens, static_cast<std::size_t>(sizeA), "A");
    problem.b = readMatrix(tokens, static_cast<std::size_t>(sizeB), "B");

    if (tokens.hasMore()) {
        throw InputError("input has more values than M*K + K*N");
    }
    return problem;
}

void writeMatrix(const std::string &path, int m, int n, const std::vector<float> &c) {
    const std::filesystem::path target(path);
    if (target.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            throw RuntimeFailure("cannot create directory " + target.parent_path().string() + ": " +
                                 ec.message());
        }
    }

    std::FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        throw RuntimeFailure("cannot open output file: " + path);
    }

    std::fprintf(file, "%d %d\n", m, n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            std::fprintf(file, col == 0 ? "%.9g" : " %.9g",
                         static_cast<double>(c[static_cast<std::size_t>(row) * n + col]));
        }
        std::fputc('\n', file);
    }

    if (std::fclose(file) != 0) {
        throw RuntimeFailure("failed to write output file: " + path);
    }
}

} // namespace lab2
