# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Dependency: Armadillo
find_package(Armadillo REQUIRED)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -fPIC -march=native)

# Target: stp
set(STP_TARGET_NAME "stp")

# For convenience, all source files are grouped into one variable
# Note: adding single headers to the list of sources is required
# so that IDEs will notice them, but it's otherwise innocuous.

set(STP_SOURCE_FILES
    stp.h
    common/fft.cpp common/ccl.cpp common/gaussian2d.cpp
    convolution/conv_func.cpp gridder/gridder.cpp pipeline/pipeline.h sourcefind/sourcefind.cpp fixtures/fixtures.h
    visibility/visibility.cpp imager/imager.h
)

# Configures which fft implementation is used
option(WITH_ARMAFFT "Builds the stp using the armadillo FFT implementation" ON)
if(WITH_ARMAFFT)
    add_definitions ( -D_ARMAFFT )
endif()

# Create the library as static
add_library(${STP_TARGET_NAME} STATIC ${STP_SOURCE_FILES})
add_dependencies(${STP_TARGET_NAME} fftw)
target_include_directories(${STP_TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${STP_TARGET_NAME} ${ARMADILLO_LIBRARIES} fftw)

