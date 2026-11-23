/**
 * @file errors.h
 * @brief Exception types. main() turns them into exit codes.
 */

#ifndef LAB2_ERRORS_H
#define LAB2_ERRORS_H

#include <stdexcept>

namespace lab2 {

/** Bad arguments or bad input data. Exit code 2. */
class InputError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/** CUDA, cuBLAS or file system failure at run time. Exit code 1. */
class RuntimeFailure : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace lab2

#endif // LAB2_ERRORS_H
