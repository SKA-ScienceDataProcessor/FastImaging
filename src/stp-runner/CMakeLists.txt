cmake_minimum_required(VERSION 2.8.9)
project (stp-runner)

#Sets the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../build/bin/stp-runner/)

find_package( Threads )

# Sets the cnpy third party library directory
set(CNPY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy/)
set(HEADER_FILES ${CNPY_DIR}/cnpy.cpp)

include_directories(${CNPY_DIR})

# Sets the libstp header files
find_library(LIBSTP NAMES libstp.a PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../build/)

add_executable(stp-runner stp-runner.cpp loadData.cpp ${HEADER_FILES})
target_link_libraries( stp-runner ${CMAKE_THREAD_LIBS_INIT} ${LIBSTP})
