# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Dependency: Threads (required because TBB is static)
find_package(Threads REQUIRED)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors)
add_compile_options(-fstrict-aliasing -fPIC -march=native)

# Target: stp
set(STP_TARGET_NAME "stp")

# For convenience, all source files are grouped into one variable
# Note: adding single headers to the list of sources is required
# so that IDEs will notice them, but it's otherwise innocuous.
set(STP_SOURCE_FILES
    stp.h types.h
    common/fft.cpp common/ccl.cpp common/matrix_math.cpp common/matstp.h
    convolution/conv_func.cpp gridder/gridder.cpp sourcefind/sourcefind.cpp sourcefind/fitting.cpp imager/imager.cpp visibility/visibility.cpp
    # The following third-party include files are added just to be noticed by IDE
    ../third-party/armadillo/include/armadillo ../third-party/fftw/api/fftw3.h ../third-party/tbb/include/tbb/tbb.h ../third-party/openblas/cblas.h
)

# Create the library as static
add_library(${STP_TARGET_NAME} STATIC ${STP_SOURCE_FILES})
add_dependencies(${STP_TARGET_NAME} tbb fftw openblas armadillo ceres)
target_include_directories(${STP_TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# Eigen target needs to be added, otherwise ceres headers no not find eigen include directory
target_link_libraries(${STP_TARGET_NAME} tbb fftw openblas Threads::Threads armadillo ceres eigen)

