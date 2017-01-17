# --- SKA Science Data Processor ---
# Reduce - Simulated pipeline run for Slow Transients Pipeline (STP) Library

# Minimum language standard is C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# Dependency: Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_NO_DEBUG)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -march=native)

# Dependency: rapidjson
find_path(RAPIDJSON_INCLUDE_PATH rapidjson)
if(RAPIDJSON_INCLUDE_PATH-NOTFOUND)
        message(FATAL_ERROR "RapidJSON not found")
endif()

# Dependency: spdlog
find_path(SPDLOG_INCLUDE_PATH spdlog)
if(SPDLOG_INCLUDE_PATH-NOTFOUND)
        message(FATAL_ERROR "spdlog not found")
endif()

# Dependency: tclap
find_path(TCLAP_INCLUDE_PATH tclap)
if(TCLAP_INCLUDE_PATH-NOTFOUND)
        message(FATAL_ERROR "TCLAP not found")
endif()

# Target: reduce
set(REDUCE_TARGET_NAME "reduce")

# Build the target
add_executable(${REDUCE_TARGET_NAME} reduce.cpp ../auxiliary/load_data.cpp)
add_dependencies(${STP_TARGET_NAME} stp cnpy-static)
target_include_directories(${REDUCE_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/ ${ARMADILLO_INCLUDE_DIRS} ${RAPIDJSON_INCLUDE_PATH}, ${TCLAP_INCLUDE_PATH} ${SPDLOG_INCLUDE_PATH})
target_link_libraries(${REDUCE_TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} stp cnpy-static ${ARMADILLO_LIBRARIES})
