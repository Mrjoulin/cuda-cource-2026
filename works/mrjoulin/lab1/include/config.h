/**
 * @file config.h
 * @brief CLI argument parsing for the submission program.
 */

#ifndef LAB1_CONFIG_H
#define LAB1_CONFIG_H

#include <string>

#include "kernels.h"

namespace lab1 {

// Exit codes required by the course submission contract:
// 0 - success, 1 - runtime error, 2 - invalid arguments or input data.
constexpr int kExitOk = 0;
constexpr int kExitRuntimeError = 1;
constexpr int kExitUsageError = 2;

/** Program settings read from the command line. */
struct Config {
  std::string inputPath;   // path to the input file with vectors
  std::string outputPath;  // path to the result file
  AccessMode mode = AccessMode::Coalesced;
};

/** Print usage help to stdout. Describes --input, --output and --mode. */
void printUsage(const char* program);

/** Parse command-line arguments into a Config. Exits on invalid arguments. */
Config parseArgs(int argc, char** argv);

}  // namespace lab1

#endif  // LAB1_CONFIG_H
