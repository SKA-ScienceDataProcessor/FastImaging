# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Python Bindings

# Dependency: Armadillo
find_package(Armadillo REQUIRED)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DARMA_NO_DEBUG)
endif()

add_definitions(-DARMA_USE_TBB_ALLOC)
add_definitions(-DARMA_DONT_USE_WRAPPER)

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

pybind11_add_module(stp_python stp_python.cpp)
target_link_libraries(stp_python stp ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
