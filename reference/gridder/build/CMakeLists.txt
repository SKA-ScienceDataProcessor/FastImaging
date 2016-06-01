cmake_minimum_required(VERSION 2.8.12)
project(gridder)

enable_testing()

# Dependencies
find_library(LIBFFTW3 fftw3)
find_package(Threads REQUIRED) # libbenchmark needs this

# Build options
add_compile_options(-Wall -O3 -march=native)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../input-data)

# Build target
set(GRIDDER_SOURCE_FILES ../src/cppGridderPrototype.cpp ../src/phasecorrection.cpp)
include_directories(../src ../../../third-party/benchmark/include)
add_executable(${PROJECT_NAME} ${GRIDDER_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBFFTW3})

# Build tests
set(TEST_SOURCE_FILES ../src/gridder_performance_test.cpp)
add_executable(gridder_performance_test ${TEST_SOURCE_FILES} ${GRIDDER_SOURCE_FILES})
set_property(TARGET gridder_performance_test PROPERTY CXX_STANDARD 11)

# TODO
target_compile_definitions(gridder_performance_test PRIVATE -DTESTING )
set(LIBBENCHMARK ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/benchmark/src/libbenchmark.a)
target_link_libraries(gridder_performance_test ${LIBBENCHMARK} ${CMAKE_THREAD_LIBS_INIT} ${LIBFFTW3})

# Tests
add_test(NAME performance COMMAND gridder_performance_test)

