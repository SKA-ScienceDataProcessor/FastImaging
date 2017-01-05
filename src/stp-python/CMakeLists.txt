# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Python Bindings

pybind11_add_module(stp_python stp_python.cpp)
target_link_libraries(stp_python stp)
