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
option(USE_GLIBCXX_PARALLEL "Uses GLIBCXX parallel mode" ON)
option(USE_FLOAT "Builds STP using 'float' type to represent real numbers" ON)
option(WITH_FUNCTION_TIMINGS "Measures function execution times from the reduce executable" ON)
option(USE_SERIAL_GRIDDER "Uses serial implementation of gridder" OFF)
option(USE_FFTSHIFT "Explicitly shifts the image and beam matrices in memory after FFT (results in slower imager)" OFF)

# Check libstdc++ parallel mode
# This needs to be done here, otherwise reduce, tests and benchmark fail linking
if(USE_GLIBCXX_PARALLEL)
    INCLUDE (CheckIncludeFileCXX)
    CHECK_INCLUDE_FILE_CXX("parallel/parallel.h" HAS_GLIBCXX_PARALLEL)
    if(HAS_GLIBCXX_PARALLEL)
        # Dependency: OPENMP
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
            # Enable libstdc++ parallel mode
            add_definitions(-D_GLIBCXX_PARALLEL)
            message(STATUS "libstdc++ parallel mode enabled")
        else()
            message(STATUS "OPENMP not found. libstdc++ parallel mode disabled")
        endif()
    else()
        message(STATUS "libstdc++ parallel mode not available")
    endif()
else()
    message(STATUS "libstdc++ parallel mode disabled")
endif()

# If ON, use float rather than double to represent real numbers (less memory usage)
if(USE_FLOAT)
    add_definitions(-DUSE_FLOAT)
endif()

# Measure function execution times from the reduce executable
if(WITH_FUNCTION_TIMINGS)
    add_definitions(-DFUNCTION_TIMINGS)
endif()

# Use serial implementation of gridder
if(USE_SERIAL_GRIDDER)
    add_definitions(-DSERIAL_GRIDDER)
endif()

# Explicitly shifts the image and beam matrices in memory after FFT (results in slower imager)
if(USE_FFTSHIFT)
    add_definitions(-DFFTSHIFT)
endif()

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

# If doxygen is available, add a target for it.
# Note: this target must be run on demand and won't be part of "all"
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYFILE ${CMAKE_CURRENT_SOURCE_DIR}/../doc/Doxyfile)
    add_custom_target(
        doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../doc
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else()
    message(WARNING "doxygen was not found. Target \"doc\" to build documentation will not be available.")
endif()

# If valgrind is available, add a target for it
# It will run on the test_pipeline_gaussiansinc executable
find_program(VALGRIND valgrind DOC "The valgrind executable path")
if(VALGRIND)
    set(VALGRIND_OUTFILE valgrind_output)
    # add valgrind as a custom target
    add_custom_target(valgrind
        DEPENDS test_pipeline_gaussiansinc
        COMMAND ${VALGRIND}
                --leak-check=full
                --show-reachable=no
                --show-possibly-lost=no
                --error-exitcode=0
                --track-origins=yes
                --log-file=${CMAKE_CURRENT_BINARY_DIR}/${VALGRIND_OUTFILE}.log
                # This is optional:
                #--xml=yes --xml-file=${CMAKE_CURRENT_BINARY_DIR}/${VALGRIND_OUTFILE}.xml
                ${CMAKE_CURRENT_BINARY_DIR}/test/test_pipeline_gaussiansinc
        WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Valgrind report will be saved to ${CMAKE_CURRENT_BINARY_DIR}/${VALGRIND_OUTFILE}.log"
    )
endif(VALGRIND)

# Add a target to generate FFTW plans
set(FFTWPLAN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/fftw-wisdom)
set(FFTWPLAN_SCRIPT generate_wisdom.sh)
# Script options
set(FFTWPLAN_OPTIONS -e third-party/fftw/bin)
if(USE_FLOAT)
    set(FFTWPLAN_OPTIONS ${FFTWPLAN_OPTIONS} -f)
endif()
# Output directory
set(FFTWPLAN_OUTPUTDIR "wisdomfiles")
# Create custom target for out-of-place fft plans (used in STP)
add_custom_target(fftwplans
    DEPENDS fftw
    COMMENT "Generated FFTW plans will be saved to ${CMAKE_CURRENT_BINARY_DIR}/${FFTWPLAN_OUTPUTDIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    # Copy script to build directory
    COMMAND ${CMAKE_COMMAND} -E copy ${FFTWPLAN_PATH}/${FFTWPLAN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}
    # Run command
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${FFTWPLAN_SCRIPT} ${FFTWPLAN_OPTIONS}
)

# Create custom target for in-place fft plans (just for benchmark)
set(IFFTWPLAN_OPTIONS ${FFTWPLAN_OPTIONS} -p)
add_custom_target(ifftwplans
    DEPENDS fftw
    COMMENT "Generated FFTW plans will be saved to ${CMAKE_CURRENT_BINARY_DIR}/${FFTWPLAN_OUTPUTDIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    # Copy script to build directory
    COMMAND ${CMAKE_COMMAND} -E copy ${FFTWPLAN_PATH}/${FFTWPLAN_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}
    # Run command
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${FFTWPLAN_SCRIPT} ${IFFTWPLAN_OPTIONS}
)

