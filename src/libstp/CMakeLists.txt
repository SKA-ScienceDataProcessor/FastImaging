# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Dependency: Armadillo (>= 7.500)
find_package(Armadillo 7.500 REQUIRED)

# Target: stp
set(STP_TARGET_NAME "stp")

# For convenience, group all source files into one variable
set(LIBSTP_SOURCE_FILES convolution/conv_func.cpp gridder/gridder.cpp)

# Create the library as static
add_library(${STP_TARGET_NAME} STATIC ${LIBSTP_SOURCE_FILES})
target_include_directories(${STP_TARGET_NAME} INTERFACE . ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${STP_TARGET_NAME} ${ARMADILLO_LIBRARIES})

# Build the unit tests
set(TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test)
set(TESTS_BIN_DIR "tests")
add_subdirectory(${TESTS_SOURCE_DIR} ${TESTS_BIN_DIR})

