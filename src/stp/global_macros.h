#ifndef GLOBAL_MACROS_H
#define GLOBAL_MACROS_H

#include <chrono>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <vector>

namespace stp {

// Macro for STP debug prints
#ifdef STPLIB_DEBUG_ON
#define STPLIB_DEBUG(logger, ...)                \
    if (spdlog::get(logger)) {                   \
        spdlog::get(logger)->debug(__VA_ARGS__); \
    }
#else
#define STPLIB_DEBUG(logger, ...) (void)0;
#endif

#ifdef FUNCTION_TIMINGS
#define NUM_TIME_INST 10
extern std::vector<std::chrono::high_resolution_clock::time_point> times_iv;
extern std::vector<std::chrono::high_resolution_clock::time_point> times_sf;
extern std::vector<std::chrono::high_resolution_clock::time_point> times_ccl;
extern std::vector<std::chrono::duration<double>> times_gridder;
#define TIMESTAMP_IMAGER times_iv.push_back(std::chrono::high_resolution_clock::now());
#define TIMESTAMP_SOURCEFIND times_sf.push_back(std::chrono::high_resolution_clock::now());
#define TIMESTAMP_CCL times_ccl.push_back(std::chrono::high_resolution_clock::now());
#else
#define TIMESTAMP_IMAGER
#define TIMESTAMP_SOURCEFIND
#define TIMESTAMP_CCL
#endif

// Macro to check if integer is power of two
#define ispowerof2(X) (X && !(X & (X - 1)))

// Convert between degrees and radians
#define deg2rad(X) ((X * M_PI) / 180.0)
#define rad2deg(X) ((X * 180.0) / M_PI)
}

#endif // GLOBAL_MACROS_H
