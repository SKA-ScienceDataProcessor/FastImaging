# --- SKA Science Data Processor ---
# Reduce - Simulated pipeline run for Slow Transients Pipeline (STP) Library

# Minimum language standard is C++17
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -march=native)
add_compile_options(-Wno-unused-parameter -Wno-sign-compare) # avoids a lot of warnings due to cnpy (save_npz)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_compile_options(-ffast-math -O3)
endif()

# Dependency: spdlog
set(SPDLOG_INCLUDE_PATH "../third-party/spdlog/include")

# Dependency: tclap
set(TCLAP_INCLUDE_PATH "../third-party/tclap/include")

# Target: reduce
set(REDUCE_TARGET_NAME "reduce")

# Build the reduce target
add_executable(${REDUCE_TARGET_NAME} reduce.cpp common_functions.cpp)
add_dependencies(${STP_TARGET_NAME} auxlib stp)
target_include_directories(${REDUCE_TARGET_NAME} PRIVATE ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${REDUCE_TARGET_NAME} auxlib stp)

# Target: run_imagevis
set(IMGVIS_TARGET_NAME "run_imager")

# Build the reduce target
add_executable(${IMGVIS_TARGET_NAME} run_imager.cpp common_functions.cpp)
add_dependencies(${STP_TARGET_NAME} auxlib stp)
target_include_directories(${IMGVIS_TARGET_NAME} PRIVATE ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${IMGVIS_TARGET_NAME} auxlib stp)

# Target: run_sourcefind
set(SOURCEFIND_TARGET_NAME "run_sourcefind")

# Build the reduce target
add_executable(${SOURCEFIND_TARGET_NAME} run_sourcefind.cpp common_functions.cpp)
add_dependencies(${STP_TARGET_NAME} auxlib stp)
target_include_directories(${SOURCEFIND_TARGET_NAME} PRIVATE ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${SOURCEFIND_TARGET_NAME} auxlib stp)

# Copy config files to the build directory
add_custom_command(
	TARGET reduce POST_BUILD
	COMMENT "Create symlink of test data and copy JSON config files to ${CMAKE_CURRENT_BINARY_DIR}"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_SOURCE_DIR}/../configs/reduce/
			${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/simdata_nstep10.npz
			simdata_nstep10.npz
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/fivesrcdata_wproj.npz
			fivesrcdata_wproj.npz
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/fivesrcdata_awproj.npz
			fivesrcdata_awproj.npz
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/wenssdata_wproj.npz
			wenssdata_wproj.npz
	COMMAND ${CMAKE_COMMAND} -E create_symlink
			${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/wenssdata_awproj.npz
			wenssdata_awproj.npz)


