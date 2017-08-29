# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Python Bindings

pybind11_add_module(stp_python stp_python.cpp)
target_link_libraries(stp_python stp openblas armadillo)

# Copy python test script to the build directory
add_custom_command(
    TARGET stp_python POST_BUILD
    COMMENT "Create symlink of test data and copy python test script to ${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/../scripts/python-bindings/test_python_bindings.py
            ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_SOURCE_DIR}/../test-data/pipeline-tests/simdata_nstep10.npz
            simdata_nstep10.npz)
