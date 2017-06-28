# --- SKA Science Data Processor ---
# Reduce - Simulated pipeline run for Slow Transients Pipeline (STP) Library

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -march=native)
add_compile_options(-Wno-unused-parameter -Wno-sign-compare) # avoids a lot of warnings due to cnpy (save_npz)

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include")

# Dependency: spdlog
set(SPDLOG_INCLUDE_PATH "../third-party/spdlog/include")

# Dependency: tclap
set(TCLAP_INCLUDE_PATH "../third-party/tclap/include")

# Target: reduce
set(REDUCE_TARGET_NAME "reduce")

# Build the reduce target
add_executable(${REDUCE_TARGET_NAME} reduce.cpp ../auxiliary/load_data.cpp ../auxiliary/save_data.h ../auxiliary/load_json_config.cpp ../auxiliary/save_json_sf_output.cpp)
add_dependencies(${STP_TARGET_NAME} stp cnpy-static armadillo)
target_include_directories(${REDUCE_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/ ${RAPIDJSON_INCLUDE_PATH} ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${REDUCE_TARGET_NAME} stp cnpy-static armadillo)

# Target: run_imagevis
set(IMGVIS_TARGET_NAME "run_imagevis")

# Build the reduce target
add_executable(${IMGVIS_TARGET_NAME} run_imagevis.cpp ../auxiliary/load_data.cpp ../auxiliary/save_data.h ../auxiliary/load_json_config.cpp ../auxiliary/save_json_sf_output.cpp)
add_dependencies(${STP_TARGET_NAME} stp cnpy-static)
target_include_directories(${IMGVIS_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/ ${RAPIDJSON_INCLUDE_PATH} ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${IMGVIS_TARGET_NAME} stp cnpy-static)

# Target: run_sourcefind
set(SOURCEFIND_TARGET_NAME "run_sourcefind")

# Build the reduce target
add_executable(${SOURCEFIND_TARGET_NAME} run_sourcefind.cpp ../auxiliary/load_data.cpp ../auxiliary/save_data.h ../auxiliary/load_json_config.cpp ../auxiliary/save_json_sf_output.cpp)
add_dependencies(${STP_TARGET_NAME} stp cnpy-static)
target_include_directories(${SOURCEFIND_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/ ${RAPIDJSON_INCLUDE_PATH} ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${SOURCEFIND_TARGET_NAME} stp cnpy-static)
