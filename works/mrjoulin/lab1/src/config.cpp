/**
 * @file config.cpp
 * @brief Implementation of command-line argument parsing.
 */

#include "config.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace lab1 {

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " --input PATH --output PATH [--mode MODE]\n"
      << "\n"
      << "Runs vector Add, SAXPY (y = alpha * x + y) and ReLU (y = max(0, x)) "
         "on the GPU\n"
      << "in the selected global-memory access mode (default: coalesced).\n"
      << "\n"
      << "Arguments:\n"
      << "  --input PATH    input file: first line \"N alpha\", then N lines "
         "\"x y\"\n"
      << "  --output PATH   result file: one line \"x+y  alpha*x+y  relu(x)\" "
         "per pair\n"
      << "  --mode MODE     coalesced or noncoalesced (optional, default: "
         "coalesced)\n"
      << "  --help          show this help and exit\n"
      << "\n"
      << "Exit codes: 0 - success, 1 - runtime error, 2 - invalid arguments or "
         "input.\n";
}

Config parseArgs(int argc, char** argv) {
  Config cfg;
  bool hasInput = false;
  bool hasOutput = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(kExitOk);
    }

    if (arg != "--input" && arg != "--output" && arg != "--mode") {
      std::cerr << "Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      std::exit(kExitUsageError);
    }

    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << arg << "\n";
      std::exit(kExitUsageError);
    }

    const std::string value = argv[++i];
    if (value.empty() || value.rfind("--", 0) == 0) {
      std::cerr << "Missing value for " << arg << "\n";
      std::exit(kExitUsageError);
    }

    if (arg == "--input") {
      cfg.inputPath = value;
      hasInput = true;
    } else if (arg == "--output") {
      cfg.outputPath = value;
      hasOutput = true;
    } else if (value == "coalesced") {
      cfg.mode = AccessMode::Coalesced;
    } else if (value == "noncoalesced") {
      cfg.mode = AccessMode::Noncoalesced;
    } else {
      std::cerr << "Invalid --mode value: " << value
                << " (expected coalesced or "
                << "noncoalesced)\n";
      std::exit(kExitUsageError);
    }
  }

  if (!hasInput || !hasOutput) {
    std::cerr << "Both --input and --output are required. See --help.\n";
    std::exit(kExitUsageError);
  }

  return cfg;
}

}  // namespace lab1
