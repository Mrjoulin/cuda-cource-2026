/**
 * @file matrix_io.h
 * @brief Text format for the matmul input and output files.
 *
 * Input:  "M K N", then A (M x K) and B (K x N), both row by row.
 * Output: "M N", then C (M x N) row by row.
 */

#ifndef LAB2_MATRIX_IO_H
#define LAB2_MATRIX_IO_H

#include <string>
#include <vector>

namespace lab2 {

/** One multiplication task read from the input file. */
struct Problem {
    int m = 0;
    int k = 0;
    int n = 0;
    std::vector<float> a;
    std::vector<float> b;
};

/** Read and validate an input file. Throws InputError on any problem. */
Problem readProblem(const std::string &path);

/** Write C with 9 significant digits. Creates missing parent directories. */
void writeMatrix(const std::string &path, int m, int n, const std::vector<float> &c);

} // namespace lab2

#endif // LAB2_MATRIX_IO_H
