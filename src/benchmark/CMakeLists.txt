# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Benchmarking Tests

# For convenience all link libraries required by
# the benchmark are put in one variable
set(BENCHMARK_LINK_LIBRARIES stp ${CMAKE_THREAD_LIBS_INIT} benchmark)

add_custom_target(benchmarking COMMAND run-parts .)

# Helper function for test target specification
function(add_benchmark_test benchmark_test_name benchmark_test_source)
   add_executable(${benchmark_test_name} ${benchmark_test_source})
   target_link_libraries(${benchmark_test_name} ${BENCHMARK_LINK_LIBRARIES})
endfunction()

# Benchmark Tests: ---------------------------------------------------------------------------------------------------

# Convolution
add_benchmark_test(test_conv_benchmark conv_test_benchmark.cpp)

# Imager
add_benchmark_test(test_imager_benchmark imager_test_benchmark.cpp)

# Kernel
add_benchmark_test(test_kernel_benchmark kernel_test_benchmark.cpp)

# Fixtures
add_benchmark_test(test_fixtures_benchmark fixtures_test_benchmark.cpp)

# Gridder
add_benchmark_test(test_gridder_benchmark gridder_test_benchmark.cpp)

# Visibility
add_benchmark_test(test_visibility_benchmark visibility_test_benchmark.cpp)

# SourceFind
add_benchmark_test(test_sourcefind_benchmark sourcefind_test_benchmark.cpp)
