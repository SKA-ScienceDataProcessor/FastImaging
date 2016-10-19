/**
*   @file stp-runner.h
*   Header file for the stp runner
*   Containts the includes, namespaces, function declarations and global shared variables to be used in the stp runner
*/

#ifndef STP_RUNNER_H
#define STP_RUNNER_H

//Third party includes
#include "../third-party/cnpy/cnpy.h"
#include <complex>
#include <stdlib.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/file_sinks.h>
#include <tclap/CmdLine.h>

//Libstp include
#include "../libstp/convolution/conv_func.h"
#include "../libstp/convolution/kernel_func.h"

//Namespaces
using namespace cnpy;
using namespace TCLAP;
using namespace spdlog;

//Constants
const char ok = 0;
const char error = 1;

//Logger
extern shared_ptr<logger> _logger;

//Paths
extern char* _log_path;

//Functions
//Numpy load functions
mat loadNpyArray(NpyArray*);

#endif /* STP_RUNNER_H */
