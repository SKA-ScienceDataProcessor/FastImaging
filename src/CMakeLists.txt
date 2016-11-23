# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Prototype

# Minimum version is current on Debian "Testing"
cmake_minimum_required(VERSION 3.6.2)

# Top-level project for the prototype. Language is C++
project(stp-prototype LANGUAGES CXX)

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing)

# The default build type is Debug. Other types can be specified
# by appending -D CMAKE_BUILD_TYPE=type to the CMake call, where
# type can be Debug, Release, RelWithDebInfo, or MinSizeRel.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Debug builds will be verbose unless otherwise specified.
# Note: verbose builds can be forced by appending VERBOSE=1 to the make call.
if(NOT CMAKE_VERBOSE_MAKEFILE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_VERBOSE_MAKEFILE ON)
    message(STATUS "Makefile is verbose")
endif()

# Testing must be enabled at top script level
enable_testing()

# Target: STP Library
set(LIBSTP_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libstp)
add_subdirectory(${LIBSTP_SOURCE_DIRECTORY})

# Target: STP Runner
set(RUNNER_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stp-runner)
add_subdirectory(${RUNNER_SOURCE_DIRECTORY})

# If cppcheck is available, add a target for it.
# Note: this target must be run on demand and won't be part of "all"
find_program(CPPCHECK cppcheck DOC "The cppcheck executable path")
if(CPPCHECK)
    message(STATUS "Found cppcheck: ${CPPCHECK}")
    message(STATUS "Target \"cppcheck\" is available")
    
    add_custom_target(
        cppcheck
        COMMAND ${CPPCHECK}
        --enable=warning,performance,portability,information,missingInclude
        --std=c++11
        --quiet
        --verbose
        ${LIBSTP_SOURCE_DIRECTORY}
        ${RUNNER_SOURCE_DIRECTORY}
    )
else()
    message(WARNING "cppcheck was not found. Target \"cppcheck\" will not be available.")
endif()
