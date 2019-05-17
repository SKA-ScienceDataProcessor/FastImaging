# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Python Bindings

# Minimum language standard is C++17
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-march=native)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_compile_options(-ffast-math -O3)
endif()

pybind11_add_module(stp_python stp_python.cpp)
target_link_libraries(stp_python PRIVATE stp armadillo)

# Copy python test script to the build directory
add_custom_command(
	TARGET stp_python POST_BUILD
	COMMENT "Create symlink of test data and copy python test script to ${CMAKE_CURRENT_BINARY_DIR}"
	COMMAND ${CMAKE_COMMAND} -E copy
			${CMAKE_SOURCE_DIR}/../scripts/python-bindings/test_python_bindings.py
			${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/simdata_nstep10.npz
			simdata_nstep10.npz
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/fivesrcdata_awproj.npz
			fivesrcdata_awproj.npz)
