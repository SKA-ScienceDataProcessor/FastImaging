cmake_minimum_required(VERSION 2.8.12)
project(gridder)

find_library(LIBFFTW3_DIR fftw3)

set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../input-data)
set(SOURCE_FILES ../src/cppGridderPrototype.cpp ../src/phasecorrection.cpp)
include_directories(../src)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBFFTW3_DIR})
