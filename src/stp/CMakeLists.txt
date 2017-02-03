# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Dependency: Threads (required because TBB is static)
find_package(Threads REQUIRED)

# Dependency: OpenBLAS
find_package(BLAS)
if(NOT BLAS_FOUND)
    message(FATAL_ERROR "OpenBLAS library was not found")
endif()

# Dependency: LAPACK
find_package(LAPACK)
if(NOT LAPACK_FOUND)
    message(FATAL_ERROR "LAPACK library was not found")
endif()

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors)
add_compile_options(-fstrict-aliasing -fPIC -march=native)

# Target: stp
set(STP_TARGET_NAME "stp")

# For convenience, all source files are grouped into one variable
# Note: adding single headers to the list of sources is required
# so that IDEs will notice them, but it's otherwise innocuous.
set(STP_SOURCE_FILES
    stp.h
    common/fft.cpp common/ccl.cpp common/vector_math.cpp
    convolution/conv_func.cpp gridder/gridder.cpp sourcefind/sourcefind.cpp imager/imager.h
    # The following third-party include files are added just to be noticed by IDE
    ../third-party/armadillo/include/armadillo ../third-party/fftw/api/fftw3.h ../third-party/tbb/include/tbb/tbb.h
)

# Create the library as static
add_library(${STP_TARGET_NAME} STATIC ${STP_SOURCE_FILES})
add_dependencies(${STP_TARGET_NAME} fftw fftw_threads tbb-static tbbmalloc-static)
# Note: since armadillo is being used purely as a template library, its include directory
# must be explicitly added to the stp target
target_include_directories(${STP_TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${STP_TARGET_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} tbb-static tbbmalloc-static
    fftw fftw_threads Threads::Threads)

