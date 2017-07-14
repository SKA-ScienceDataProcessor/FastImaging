# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Benchmarking Tests

set(CMAKE_CXX_STANDARD 14)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -march=native)

# Dependency: Threads (required by google benchmark)
find_package(Threads REQUIRED)

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include")

# Configures the location of the test data
get_filename_component(PIPELINE_DATAPATH "../../test-data/pipeline-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_DATAPATH="${PIPELINE_DATAPATH}/")

# Configures the location of the configuration files
get_filename_component(PIPELINE_CONFIGPATH "../../configs/reduce" ABSOLUTE)
add_definitions ( -D_PIPELINE_CONFIGPATH="${PIPELINE_CONFIGPATH}/")

# Configures the location of the fftw wisdom files
if(USE_FLOAT)
   get_filename_component(WISDOM_FILEPATH "../../scripts/fftw-wisdom/wisdomfiles_f" ABSOLUTE)
else()
   get_filename_component(WISDOM_FILEPATH "../../scripts/fftw-wisdom/wisdomfiles" ABSOLUTE)
endif()
add_definitions ( -D_WISDOM_FILEPATH="${WISDOM_FILEPATH}/")

# For convenience all link libraries required by
# the benchmark are put in one variable
set(BENCHMARK_LINK_LIBRARIES stp benchmark Threads::Threads armadillo)

add_custom_target(benchmarking COMMAND run-parts .)

# Helper function for test target specification
function(add_benchmark_test benchmark_test_name benchmark_test_source)
   add_executable(${benchmark_test_name} ${benchmark_test_source} ../auxiliary/load_data.cpp ../auxiliary/load_json_config.cpp)
   add_dependencies(${benchmark_test_name} stp cnpy-static benchmark armadillo)
   target_include_directories(${benchmark_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
       ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${RAPIDJSON_INCLUDE_PATH})
   target_link_libraries(${benchmark_test_name} ${BENCHMARK_LINK_LIBRARIES} cnpy-static)
endfunction()

# Benchmark Tests: ---------------------------------------------------------------------------------------------------

# Memset
add_benchmark_test(memset_benchmark memset_benchmark.cpp)

# Matrix Math
add_benchmark_test(matrix_math_benchmark matrix_math_benchmark.cpp)

# Median Math
add_benchmark_test(median_benchmark median_benchmark.cpp)

# Convolution_function
add_benchmark_test(conv_benchmark conv_benchmark.cpp)

# Populate_cache
add_benchmark_test(populate_kernel_cache_benchmark populate_kernel_cache_benchmark.cpp)

# FFT C2R
add_benchmark_test(fft_c2r_benchmark fft_c2r_benchmark.cpp)

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

