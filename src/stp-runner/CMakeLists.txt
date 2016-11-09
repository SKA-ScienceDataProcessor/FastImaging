# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Runner

# Dependency: cnpy
# Note: a binary directory must be specified since the third-party
# directory isn't under libstp.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
                 ${CMAKE_CURRENT_BINARY_DIR}/cnpy)

# Dependency: Libstp
set(RUNNER_LINK_LIBRARIES ${STP_TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} stp cnpy)

# Build the runner
add_executable(stp_runner stp_runner.cpp load_data.cpp)
target_include_directories(stp_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(stp_runner ${RUNNER_LINK_LIBRARIES})
