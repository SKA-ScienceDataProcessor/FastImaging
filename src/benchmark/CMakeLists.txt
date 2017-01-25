# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Benchmarking Tests

set(CMAKE_CXX_STANDARD 14)
# Dependency: Armadillo
find_package(Armadillo REQUIRED)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DARMA_NO_DEBUG)
endif()
add_definitions(-DARMA_USE_TBB_ALLOC)
add_definitions(-DARMA_DONT_USE_WRAPPER)

# Dependency: Threads (required by google benchmark)
find_package(Threads)

# Dependency: rapidjson
find_path(RAPIDJSON_INCLUDE_PATH rapidjson)
if(RAPIDJSON_INCLUDE_PATH-NOTFOUND)
        message(FATAL_ERROR "RapidJSON not found")
endif()

# Configures the location of the test data
get_filename_component(PIPELINE_DATAPATH "../../test-data/pipeline-data" ABSOLUTE)
add_definitions ( -D_PIPELINE_DATAPATH="${PIPELINE_DATAPATH}/")

# Configures the location of the configuration files
get_filename_component(PIPELINE_CONFIGPATH "../../config/pipeline-config" ABSOLUTE)
add_definitions ( -D_PIPELINE_CONFIGPATH="${PIPELINE_CONFIGPATH}/")

# For convenience all link libraries required by
# the benchmark are put in one variable
set(BENCHMARK_LINK_LIBRARIES stp benchmark ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(benchmarking COMMAND run-parts .)

# Helper function for test target specification
function(add_benchmark_test benchmark_test_name benchmark_test_source)
   add_executable(${benchmark_test_name} ${benchmark_test_source} ../auxiliary/load_data.cpp ../auxiliary/load_json_config.cpp)
   target_include_directories(${benchmark_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
       ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${ARMADILLO_INCLUDE_DIRS} ${RAPIDJSON_INCLUDE_PATH})
   target_link_libraries(${benchmark_test_name} ${BENCHMARK_LINK_LIBRARIES} cnpy-static)
endfunction()

# Benchmark Tests: ---------------------------------------------------------------------------------------------------

# Vector Math
add_benchmark_test(vector_math_benchmark vector_math_benchmark.cpp)

# Convolution
add_benchmark_test(conv_benchmark conv_benchmark.cpp)

# Populate_cache
add_benchmark_test(populate_kernel_cache_benchmark populate_kernel_cache_benchmark.cpp)

# FFT
add_benchmark_test(fft_benchmark fft_benchmark.cpp)

# Imager
add_benchmark_test(imager_benchmark imager_benchmark.cpp)

# Kernel
add_benchmark_test(kernel_benchmark kernel_benchmark.cpp)

# Gridder
add_benchmark_test(gridder_benchmark gridder_benchmark.cpp)

# SourceFind
add_benchmark_test(sourcefind_benchmark sourcefind_benchmark.cpp)

# Pipeline
add_benchmark_test(pipeline_benchmark pipeline_benchmark.cpp)

