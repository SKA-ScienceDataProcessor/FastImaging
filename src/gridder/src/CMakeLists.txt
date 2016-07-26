cmake_minimum_required(VERSION 2.8.9)
project (gridding)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)

# Find the FFTW v3 dependency
find_library(LIBFFTW3_DIR fftw3)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Create the Phase Correction library 
add_library(phasecorrection STATIC phasecorrection.cpp)

# And then the gridder prototype
add_executable(cppGridderPrototype cppGridderPrototype.cpp gridderprototype.cpp)

# Finally link to the libraries
target_link_libraries(cppGridderPrototype phasecorrection ${LIBFFTW3_DIR} ${ARMADILLO_LIBRARIES})
add_definitions(-std=c++11)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3 -march=native")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()



