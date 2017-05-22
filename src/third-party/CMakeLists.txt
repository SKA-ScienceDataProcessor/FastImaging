# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Prototype Dependencies (in-tree)

# Dependency: Google Test
set(BUILD_GTEST ON CACHE BOOL "Builds the googletest subproject")
set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject")
add_subdirectory(googletest)

# Dependency: Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
add_subdirectory(benchmark)

# Dependency: Cnpy
# ZLIB is a dependency of cnpy which is missing in cnpy's CMakefile
find_package(ZLIB)
if(NOT ZLIB_FOUND)
    message(FATAL_ERROR "ZLIB library was not found")
endif()
# Add cnpy subdirectory
add_subdirectory(cnpy)
# The cnpy CMake script does not export the interface include directory, so
# it must be explicitly specified here.
set_target_properties(cnpy-static PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

# Dependency: pybind11 (1.8.1)
add_subdirectory(pybind11)

# Dependency: FFTW3 (3.3.5)
set(FFTW_TARGET_NAME fftw)
set(FFTW_PROJECT_NAME ${FFTW_TARGET_NAME}_project)
set(FFTW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${FFTW_TARGET_NAME})
set(FFTW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${FFTW_TARGET_NAME})

# Build from source using the classical approach of configure & make.
# Note: generation of position independent code (PIC) is explicitly requested
#       so that shared libraries (e.g. STP Python) can link agains the library.
include(ExternalProject)

# Set FFTW configure flags
if(USE_FLOAT)
    message(STATUS "FFTW: use --enable-float")
    set(FFTW_CONF_FLAGS --with-pic --enable-static --disable-doc --disable-fortran
        --enable-threads --enable-sse2 --enable-avx --enable-avx2 --enable-float)
else()
    message(STATUS "FFTW: do not use --enable-float")
    set(FFTW_CONF_FLAGS --with-pic --enable-static --disable-doc --disable-fortran
        --enable-threads --enable-sse2 --enable-avx --enable-avx2)
endif()
# Other available options: --enable-avx512 --enable-avx-128-fma --enable-generic-simd128 --enable-generic-simd256

ExternalProject_Add(
    ${FFTW_PROJECT_NAME}
    SOURCE_DIR ${FFTW_SOURCE_DIR}
    CONFIGURE_COMMAND ./configure --prefix=${FFTW_INSTALL_DIR} ${FFTW_CONF_FLAGS}
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install && make distclean
)

# Specify the target for the external library
add_library(${FFTW_TARGET_NAME} INTERFACE)
if(USE_FLOAT)
    target_link_libraries(${FFTW_TARGET_NAME} INTERFACE ${FFTW_INSTALL_DIR}/lib/libfftw3f.a ${FFTW_INSTALL_DIR}/lib/libfftw3f_threads.a)
else()
    target_link_libraries(${FFTW_TARGET_NAME} INTERFACE ${FFTW_INSTALL_DIR}/lib/libfftw3.a ${FFTW_INSTALL_DIR}/lib/libfftw3_threads.a)
endif()
target_include_directories(${FFTW_TARGET_NAME} INTERFACE ${FFTW_INSTALL_DIR}/include)
add_dependencies(${FFTW_TARGET_NAME} ${FFTW_PROJECT_NAME})


# Dependency: TBB (2017 Update3)
set(TBB_TARGET_NAME tbb)
set(TBB_PROJECT_NAME ${TBB_TARGET_NAME}_project)
set(TBB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${TBB_TARGET_NAME})
set(TBB_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TBB_TARGET_NAME})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_BUILD_DIR ${TBB_SOURCE_DIR}/build/${TBB_TARGET_NAME}_debug)
else()
    set(TBB_BUILD_DIR ${TBB_SOURCE_DIR}/build/${TBB_TARGET_NAME}_release)
endif()

# Build from source using the classical approach of make.
include(ExternalProject)
ExternalProject_Add(
    ${TBB_PROJECT_NAME}
    SOURCE_DIR ${TBB_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make tbb_build_prefix=${TBB_TARGET_NAME} extra_inc=big_iron.inc
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND cp -r ${TBB_BUILD_DIR} ${TBB_INSTALL_DIR} &&
      cp -r ${TBB_SOURCE_DIR}/include ${TBB_INSTALL_DIR} && make tbb_build_prefix=${TBB_TARGET_NAME} clean
)

# Specify the target for the external library
add_library(${TBB_TARGET_NAME} INTERFACE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(${TBB_TARGET_NAME} INTERFACE ${TBB_INSTALL_DIR}/libtbb_debug.a ${TBB_INSTALL_DIR}/libtbbmalloc_debug.a)
else()
    target_link_libraries(${TBB_TARGET_NAME} INTERFACE ${TBB_INSTALL_DIR}/libtbb.a ${TBB_INSTALL_DIR}/libtbbmalloc.a)
endif()
target_include_directories(${TBB_TARGET_NAME} INTERFACE ${TBB_INSTALL_DIR}/include)
add_dependencies(${TBB_TARGET_NAME} ${TBB_PROJECT_NAME})


# Dependency: OpenBLAS (0.2.19)
set(OPENBLAS_TARGET_NAME openblas)
set(OPENBLAS_PROJECT_NAME ${OPENBLAS_TARGET_NAME}_project)
set(OPENBLAS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${OPENBLAS_TARGET_NAME})
set(OPENBLAS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${OPENBLAS_TARGET_NAME})

# Build from source using the classical approach of make.
include(ExternalProject)
ExternalProject_Add(
    ${OPENBLAS_PROJECT_NAME}
    SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install PREFIX=${OPENBLAS_INSTALL_DIR} && make clean
)

# Specify the target for the external library
add_library(${OPENBLAS_TARGET_NAME} INTERFACE)
target_link_libraries(${OPENBLAS_TARGET_NAME} INTERFACE ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a)
target_include_directories(${OPENBLAS_TARGET_NAME} INTERFACE ${OPENBLAS_INSTALL_DIR}/include)
add_dependencies(${OPENBLAS_TARGET_NAME} ${OPENBLAS_PROJECT_NAME})

