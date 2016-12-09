# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Dependency: Armadillo (>= 7.500)
find_package(Armadillo 7.500 REQUIRED)

# Dependency: FFTW3 (3.3.5)
set(FFTW_TARGET_NAME fftw)
set(FFTW_PROJECT_NAME ${FFTW_TARGET_NAME}_project)
set(FFTW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/${FFTW_TARGET_NAME})
set(FFTW_INCLUDE_DIR ${FFTW_SOURCE_DIR}/api)
set(FFTW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${FFTW_TARGET_NAME})

include(ExternalProject)
ExternalProject_Add(
    ${FFTW_PROJECT_NAME}
    SOURCE_DIR ${FFTW_SOURCE_DIR}
    CONFIGURE_COMMAND ./configure --prefix=${FFTW_INSTALL_DIR}
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install && make distclean
)

add_library(${FFTW_TARGET_NAME} STATIC IMPORTED)
set_property(TARGET ${FFTW_TARGET_NAME} PROPERTY IMPORTED_LOCATION ${FFTW_INSTALL_DIR}/lib/libfftw3.a)
add_dependencies(${FFTW_TARGET_NAME} ${FFTW_PROJECT_NAME})

# Target: stp
set(STP_TARGET_NAME "stp")

# For convenience, all source files are grouped into one variable
# Note: adding single headers to the list of sources is required
# so that IDEs will notice them, but it's otherwise innocuous.
set(LIBSTP_SOURCE_FILES libstp.h convolution/conv_func.cpp gridder/gridder.cpp pipeline/pipeline.h sourcefind/sourcefind.cpp fixtures/fixtures.h common/fft.cpp common/ccl.cpp common/gaussian2d.cpp)

# Configures the pipeline.h on which fft implementation to use

option(ARMA_IMP "Builds the libstp using the armadillo FFT implementation" ON)

if(ARMA_IMP)
    add_definitions ( -D_ARMA=0 )
else()
    add_definitions ( -D_ARMA=1 )
endif()
configure_file(pipeline/pipeline.h pipeline/pipeline.h @ONLY)

# Create the library as static
add_library(${STP_TARGET_NAME} STATIC ${LIBSTP_SOURCE_FILES})
target_include_directories(${STP_TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${ARMADILLO_INCLUDE_DIRS} PRIVATE ${FFTW_INCLUDE_DIR})
target_link_libraries(${STP_TARGET_NAME} ${ARMADILLO_LIBRARIES} ${FFTW_TARGET_NAME})

# Build the unit tests
set(TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test)
set(TESTS_BIN_DIR "tests")
add_subdirectory(${TESTS_SOURCE_DIR} ${TESTS_BIN_DIR})

