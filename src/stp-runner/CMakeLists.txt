# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Runner

# Dependency: rapidjson
find_path(RAPIDJSON rapidjson)
if(${RAPIDJSON} STREQUAL "RAPIDJSON-NOTFOUND")
	message(FATAL_ERROR "rapidjson not found")
endif()

# Dependency: Libstp
set(RUNNER_LINK_LIBRARIES ${STP_TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} stp cnpy)

# Dependency: auxiliary
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/../auxiliary)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_BINARY_DIR}/../auxiliary)
endif()

# Copy configuration files to the binary directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configuration_data/configuration.json ${CMAKE_CURRENT_BINARY_DIR}/configuration.json COPYONLY)

# Build the runner
add_executable(stp_runner stp_runner.cpp)
target_include_directories(stp_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary/)
target_link_libraries(stp_runner ${RUNNER_LINK_LIBRARIES} auxiliary cnpy)
