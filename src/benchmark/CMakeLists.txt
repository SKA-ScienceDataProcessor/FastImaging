# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Benchmarking Tests

# Dependency: Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_NO_DEBUG)

# Dependency: Threads (required by google benchmark)
find_package(Threads)

# Configures the location of the test data
get_filename_component(TDATA_PATH "../../test-data" ABSOLUTE)
add_definitions ( -D_TESTPATH="${TDATA_PATH}/")

# For convenience all link libraries required by
# the benchmark are put in one variable
set(BENCHMARK_LINK_LIBRARIES stp benchmark ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(benchmarking COMMAND run-parts .)

# Helper function for test target specification
function(add_benchmark_test benchmark_test_name benchmark_test_source)
   add_executable(${benchmark_test_name} ${benchmark_test_source} ../auxiliary/load_data.cpp)
   target_include_directories(${benchmark_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy
       ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${ARMADILLO_INCLUDE_DIRS})
   target_link_libraries(${benchmark_test_name} ${BENCHMARK_LINK_LIBRARIES} cnpy-static)
endfunction()

# Benchmark Tests: ---------------------------------------------------------------------------------------------------

# Convolution
add_benchmark_test(conv_benchmark conv_benchmark.cpp)

# Populate_cache
add_benchmark_test(populate_kernel_cache_benchmark populate_kernel_cache_benchmark.cpp)

# Imager
add_benchmark_test(imager_benchmark imager_benchmark.cpp)

# Kernel
add_benchmark_test(kernel_benchmark kernel_benchmark.cpp)

# Fixtures
add_benchmark_test(fixtures_benchmark fixtures_benchmark.cpp)

# Gridder
add_benchmark_test(gridder_benchmark gridder_benchmark.cpp)

# Visibility
add_benchmark_test(visibility_benchmark visibility_benchmark.cpp)

# SourceFind
add_benchmark_test(sourcefind_benchmark sourcefind_benchmark.cpp)

# Pipeline
add_benchmark_test(pipeline_benchmark pipeline_benchmark.cpp)

