#include "config.h"
#include "kernels.h"

#include <cstdlib>
#include <iostream>
#include <string>


namespace lab1 {

namespace {

// Returns the value that follows the option at argv[*index] and advances the
// index past it. Exits when the value is missing.
std::string ReadOptionValue(int argc, char** argv, int* index) {
  const std::string arg = argv[*index];

  if (*index + 1 >= argc) {
    std::cerr << "Missing value for " << arg << "\n";
    std::exit(exit_usage_error);
  }

  const std::string value = argv[++*index];
  if (value.empty() || value.rfind("--", 0) == 0) {
    std::cerr << "Missing value for " << arg << "\n";
    std::exit(exit_usage_error);
  }
  return value;
}

AccessMode ParseMode(const std::string& value) {
  if (value == "coalesced") {
    return AccessMode::Coalesced;
  }
  if (value == "noncoalesced") {
    return AccessMode::Noncoalesced;
  }
  std::cerr << "Invalid --mode value: " << value << " (expected coalesced or "
            << "noncoalesced)\n";
  std::exit(exit_usage_error);
}

}  // namespace

void PrintUsage(const char* program) {
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

Config ParseArgs(int argc, char** argv) {
  Config cfg;
  bool has_input = false;
  bool has_output = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(exit_ok);
    }

    if (arg != "--input" && arg != "--output" && arg != "--mode") {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      std::exit(exit_usage_error);
    }

    const std::string value = ReadOptionValue(argc, argv, &i);
    if (arg == "--input") {
      cfg.input_path = value;
      has_input = true;
    } else if (arg == "--output") {
      cfg.output_path = value;
      has_output = true;
    } else {
      cfg.mode = ParseMode(value);
    }
  }

  if (!has_input || !has_output) {
    std::cerr << "Both --input and --output are required. See --help.\n";
    std::exit(exit_usage_error);
  }

  return cfg;
}

}  // namespace lab1
