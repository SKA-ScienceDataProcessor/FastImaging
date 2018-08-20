#ifndef GLOBAL_MACROS_H
#define GLOBAL_MACROS_H

#include <chrono>
#include <vector>

namespace stp {

// Macro for STP debug prints
#ifdef STPLIB_DEBUG_ON
#include <spdlog/spdlog.h>
#define STPLIB_DEBUG(logger, ...)   \
    if (logger) {                   \
        logger->debug(__VA_ARGS__); \
    }
#else
#define STPLIB_DEBUG(logger, ...) (void)0;
#endif

#ifdef FUNCTION_TIMINGS
#define NUM_TIME_INST 10
extern std::vector<std::chrono::high_resolution_clock::time_point> times_iv;
extern std::vector<std::chrono::high_resolution_clock::time_point> times_sf;
extern std::vector<std::chrono::high_resolution_clock::time_point> times_ccl;
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
}

#endif // GLOBAL_MACROS_H
