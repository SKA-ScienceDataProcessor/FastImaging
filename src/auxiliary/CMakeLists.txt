# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library

# Minimum language standard is C++17
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -fPIC -march=native)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-ffast-math -O3) # -fopt-info-vec-missed -fopt-info-vec-optimized)
endif()

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include")

# Target: auxiliary
set(AUX_TARGET_NAME "auxlib")

# For convenience, all source files are grouped into one variable
# Note: adding single headers to the list of sources is required
# so that IDEs will notice them, but it's otherwise innocuous.
set(AUX_SOURCE_FILES
../auxiliary/fixtures.h ../auxiliary/gaussian2d.cpp ../auxiliary/linear_interpolation.cpp
../auxiliary/load_data.cpp ../auxiliary/save_data.h ../auxiliary/load_json_config.cpp ../auxiliary/save_json_sf_output.cpp
)

# Create the library as static
add_library(${AUX_TARGET_NAME} STATIC ${AUX_SOURCE_FILES})
add_dependencies(${AUX_TARGET_NAME} cnpy-stp stp)
target_include_directories(${AUX_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${RAPIDJSON_INCLUDE_PATH})
target_link_libraries(${AUX_TARGET_NAME} cnpy-stp stp)

