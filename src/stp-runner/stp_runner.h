/**
*   @file stp_runner.h
*   Header file for the stp runner
*   Containts the includes, namespaces, function declarations and global shared variables to be used in the stp runner
*/

#ifndef STP_RUNNER_H
#define STP_RUNNER_H

#include <libstp.h>

#include <stdlib.h>
#include <fstream>
#include <complex>

#include <cnpy.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/file_sinks.h>

#include <tclap/CmdLine.h>

//Constants
const char ok(0);
const char error(1);

//Function types
enum func {
    tophat,
    triangle,
    sinc,
    gaussian,
    gaussianSinc,
    kernel_tophat,
    kernel_triangle
};

//Logger
extern std::shared_ptr<spdlog::logger> _logger;

//Paths
extern char* _log_path;

//Functions
//Numpy load functions
arma::mat loadNpyArray(cnpy::NpyArray*);

#endif /* STP_RUNNER_H */
