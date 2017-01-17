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
add_subdirectory(cnpy)
# The cnpy CMake script does not export the interface include directory, so
# it must be explicitly specified here.
set_target_properties(cnpy-static PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

# Dependency: pybind11 (1.8.1)
add_subdirectory(pybind11)

# Dependency: FFTW3 (3.3.5)
set(FFTW_TARGET_NAME fftw)
set(FFTW_THREADS_TARGET_NAME fftw_threads)
set(FFTW_PROJECT_NAME ${FFTW_TARGET_NAME}_project)
set(FFTW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${FFTW_TARGET_NAME})
set(FFTW_INCLUDE_DIR ${FFTW_SOURCE_DIR}/api)
set(FFTW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${FFTW_TARGET_NAME})

# Build from source using the classical approach of configure & make.
# Note: generation of position independent code (PIC) is explicitly requested
#       so that shared libraries (e.g. STP Python) can link agains the library.
include(ExternalProject)
ExternalProject_Add(
    ${FFTW_PROJECT_NAME}
    SOURCE_DIR ${FFTW_SOURCE_DIR}
    CONFIGURE_COMMAND ./configure --prefix=${FFTW_INSTALL_DIR} --with-pic --enable-threads
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install && make distclean
)

# Specify the target for the external library
# Note: GLOBAL must be used to extend the visibility of imported libraries
add_library(${FFTW_TARGET_NAME} STATIC IMPORTED GLOBAL)
add_library(${FFTW_THREADS_TARGET_NAME} STATIC IMPORTED GLOBAL)
set_property(TARGET ${FFTW_TARGET_NAME} PROPERTY IMPORTED_LOCATION ${FFTW_INSTALL_DIR}/lib/libfftw3.a)
set_property(TARGET ${FFTW_THREADS_TARGET_NAME} PROPERTY IMPORTED_LOCATION ${FFTW_INSTALL_DIR}/lib/libfftw3_threads.a)
add_dependencies(${FFTW_TARGET_NAME} ${FFTW_PROJECT_NAME})
add_dependencies(${FFTW_THREADS_TARGET_NAME} ${FFTW_PROJECT_NAME})
# Make sure targets linking against the library get the include directory
set_target_properties(${FFTW_TARGET_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR})
set_target_properties(${FFTW_THREADS_TARGET_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR})

# Dependency: TBB (2017 Update3)
set(TBB_TARGET_NAME tbb)
set(TBB_PROJECT_NAME ${TBB_TARGET_NAME}_project)
set(TBB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${TBB_TARGET_NAME})
set(TBB_INCLUDE_DIR ${TBB_SOURCE_DIR}/include)
set(TBB_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TBB_TARGET_NAME})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_BUILD_DIR ${TBB_SOURCE_DIR}/build/${TBB_TARGET_NAME}_debug)
    set(TBB_LIBS ${TBB_INSTALL_DIR}/libtbb_debug.a)
else()
    set(TBB_BUILD_DIR ${TBB_SOURCE_DIR}/build/${TBB_TARGET_NAME}_release)
    set(TBB_LIBS ${TBB_INSTALL_DIR}/libtbb.a)
endif()

# Build from source using the classical approach of make.
include(ExternalProject)
ExternalProject_Add(
    ${TBB_PROJECT_NAME}
    SOURCE_DIR ${TBB_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make tbb_build_prefix=${TBB_TARGET_NAME} extra_inc=big_iron.inc
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND cp -r ${TBB_BUILD_DIR} ${TBB_INSTALL_DIR} && make tbb_build_prefix=${TBB_TARGET_NAME} clean
)

# Specify the target for the external library
# Note: GLOBAL must be used to extend the visibility of imported libraries
add_library(${TBB_TARGET_NAME} STATIC IMPORTED GLOBAL)
set_property(TARGET ${TBB_TARGET_NAME} PROPERTY IMPORTED_LOCATION ${TBB_LIBS})
add_dependencies(${TBB_TARGET_NAME} ${TBB_PROJECT_NAME})
# Make sure targets linking against the library get the include directory
set_target_properties(${TBB_TARGET_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIR})

