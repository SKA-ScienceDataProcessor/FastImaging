#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include "stp.h"

#include <load_json_config.h>
#include <spdlog/spdlog.h>
#include <tclap/CmdLine.h>

#ifdef FUNCTION_TIMINGS
#define TIMESTAMP_MAIN times_main.push_back(std::chrono::high_resolution_clock::now());
#else
TIMESTAMP_MAIN
#endif

// Loggers
extern std::shared_ptr<spdlog::logger> reducelogger;
extern std::shared_ptr<spdlog::logger> srclogger;
extern std::shared_ptr<spdlog::logger> benchlogger;
extern TCLAP::MultiSwitchArg enableLoggerArg;

/**
* @brief Logger initialization function
*
* Creates and initializes the logger to be used throughout the program
* @param[in] logging_value (int): Defines logging level (0 to disable).
*/
void initLogger(int logging_value);

/**
* @brief Check if image size is power of two. If not, it changes image_size.
*
* @param[in] image_size (int): Image size value.
*/
void check_image_size(int& image_size);

/**
* @brief Log imager configuration parameters.
*
* @param[in] cfg (ConfigurationFile): Configuration file values.
*/
void log_configuration_imager(const ConfigurationFile& cfg);

/**
* @brief Log sourcefind configuration parameters.
*
* @param[in] cfg (ConfigurationFile): Configuration file values.
*/
void log_configuration_sourcefind(const ConfigurationFile& cfg);

/**
* @brief Log parameters of each detected island
*
* @param[in] sfimage (SourceFindImage): Source find struct with the list of detected islands.
*/
void log_detected_islands(stp::SourceFindImage& sfimage, bool print_gaussian_fit);

/**
* @brief Log function timings (accessible from global variables).
*/
void log_function_timings();

#endif // COMMON_FUNCTIONS_H
