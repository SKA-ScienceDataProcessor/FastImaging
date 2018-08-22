# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Benchmarking Tests

# Minimum language standard is C++17
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-Wall -Wextra -Wfloat-equal -pedantic -pedantic-errors -fstrict-aliasing -march=native)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-ffast-math -O3)
endif()

# Dependency: Threads (required by google benchmark)
find_package(Threads REQUIRED)

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include")

# Configures the location of the test data
get_filename_component(PIPELINE_DATAPATH "../../test-data/pipeline-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_DATAPATH="${PIPELINE_DATAPATH}/")

# Configures the location of the configuration files
get_filename_component(PIPELINE_CONFIGPATH "../../configs/benchmark-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_CONFIGPATH="${PIPELINE_CONFIGPATH}/")

# For convenience all link libraries required by
# the benchmark are put in one variable
set(BENCHMARK_LINK_LIBRARIES auxlib stp benchmark Threads::Threads armadillo)

add_custom_target(benchmarking COMMAND run-parts .)

# Helper function for test target specification
function(add_benchmark_test benchmark_test_name benchmark_test_source)
   add_executable(${benchmark_test_name} ${benchmark_test_source})
   add_dependencies(${benchmark_test_name} auxlib stp benchmark armadillo)
   target_link_libraries(${benchmark_test_name} ${BENCHMARK_LINK_LIBRARIES})
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

# FFT shift
add_benchmark_test(fft_shift_benchmark fftshift_benchmark.cpp)

# Imager
add_benchmark_test(imager_benchmark imager_benchmark.cpp)

# Interpolation
add_benchmark_test(interpolation_benchmark interpolation_benchmark.cpp)

# Kernel
add_benchmark_test(kernel_benchmark kernel_benchmark.cpp)

# Gridder
add_benchmark_test(gridder_benchmark gridder_benchmark.cpp)

# SourceFind
add_benchmark_test(sourcefind_benchmark sourcefind_benchmark.cpp)

# Pipeline
add_benchmark_test(pipeline_benchmark pipeline_benchmark.cpp)
