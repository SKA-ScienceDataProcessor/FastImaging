# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Prototype

# Minimum version is the one shipping with Ubuntu 16.04 LTS
cmake_minimum_required(VERSION 3.5.1)

# Top-level project for the prototype.
project(stp-prototype)

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

# Options
option(BUILD_TESTS "Builds the unit tests" ON)
option(BUILD_BENCHMARK "Builds the benchmark tests" ON)

# Sub-project: third-party dependencies (Google Test, Google Benchmark)
set(THIRD_PARTY_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third-party)
add_subdirectory(${THIRD_PARTY_SOURCE_DIRECTORY})

# Sub-project: STP Library
set(STP_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stp)
add_subdirectory(${STP_SOURCE_DIRECTORY})

# Target: STP Reduce
set(REDUCE_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reduce)
add_subdirectory(${REDUCE_SOURCE_DIRECTORY})

# Sub-project: Python Bindings
set(PYTHON_BINDINGS_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stp-python)
add_subdirectory(${PYTHON_BINDINGS_SOURCE_DIRECTORY})

# Sub-project: unit tests
if(BUILD_TESTS)
    # Note: testing must be enabled at top script level
    enable_testing()
    set(TESTS_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
    add_subdirectory(${TESTS_SOURCE_DIRECTORY})
else()
    message(STATUS "Unit tests will not be built")
endif()

# Sub-project: benchmarking tests
if(BUILD_BENCHMARK)
    set(BENCHMARK_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark)
    add_subdirectory(${BENCHMARK_SOURCE_DIRECTORY})
else()
    message(STATUS "Benchmarking tests will not be built")
endif()

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
        ${STP_SOURCE_DIRECTORY}
    )
else()
    message(WARNING "cppcheck was not found. Target \"cppcheck\" will not be available.")
endif()

