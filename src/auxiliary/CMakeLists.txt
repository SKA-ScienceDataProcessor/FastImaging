# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Auxiliary functions

# Dependency: Armadillo (>= 7.500)
find_package(Armadillo 7.500 REQUIRED)

# Dependency: cnpy
# Note: a binary directory must be specified since the third-party
# directory isn't under libstp.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
                 ${CMAKE_CURRENT_BINARY_DIR}/cnpy)

# Build the auxiliary functions
add_library(auxiliary load_data.cpp load_data.h)
target_include_directories(auxiliary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(auxiliary cnpy)
