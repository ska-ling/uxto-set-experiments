#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>

#define FMT_HEADER_ONLY 1
#include <fmt/core.h>

// Declarar la variable como extern (solo declaración, no definición)
extern std::ofstream log_file;

// Custom logging function that writes to both stdout and log file
template<typename... Args>
void log_print(fmt::format_string<Args...> fmt, Args&&... args) {
    std::string msg = fmt::format(fmt, std::forward<Args>(args)...);
    fmt::print("{}", msg);  // Print to stdout
    if (log_file.is_open()) {
        log_file << msg;    // Write to log file
        log_file.flush();   // Ensure it's written immediately
    }
}

// Initialize the log file
void init_log_file(const std::string& benchmark_name);

// Close the log file
void close_log_file();
