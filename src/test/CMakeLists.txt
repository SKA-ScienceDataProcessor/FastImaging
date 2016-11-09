# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Unit Tests

# Note: a binary directory must be specified for the dependencies
# since the third-party directory isn't under libstp.

# Dependency: Google Test
set(BUILD_GTEST ON CACHE BOOL "Builds the googletest subproject")
set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/googletest
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest)

# Dependency: Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/benchmark
                 ${CMAKE_CURRENT_BINARY_DIR}/benchmark)

# TODO: Dependency: OpenBLAS
# find_package(BLAS REQUIRED)

# For convenience all link libraries required by
# the tests are put in one variable
set(TEST_LINK_LIBRARIES ${STP_TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main benchmark)

# Convolution Functions Tests

# Triangle
add_executable(test_conv_triangle_func conv_func/conv_func_triangle.cpp)
target_link_libraries(test_conv_triangle_func ${TEST_LINK_LIBRARIES})

# Tophat
add_executable(test_conv_tophat conv_func/conv_func_tophat.cpp)
target_link_libraries(test_conv_tophat ${TEST_LINK_LIBRARIES})

# Sinc
add_executable(test_conv_sinc conv_func/conv_func_sinc.cpp)
target_link_libraries(test_conv_sinc ${TEST_LINK_LIBRARIES})

# Gaussian
add_executable(test_conv_gaussian conv_func/conv_func_gaussian.cpp)
target_link_libraries(test_conv_gaussian ${TEST_LINK_LIBRARIES})

# Gaussian Sinc
add_executable(test_conv_gaussian_sinc conv_func/conv_func_gaussiansinc.cpp)
target_link_libraries(test_conv_gaussian_sinc ${TEST_LINK_LIBRARIES})


# Kernel Functions Tests

# Oversampled Pillbox
add_executable(test_kernel_generation_oversampled_pillbox kernel_func/kernel_func_OversampledPillbox.cpp)
target_link_libraries(test_kernel_generation_oversampled_pillbox ${TEST_LINK_LIBRARIES})

# Oversampled Pillbox Small
add_executable(test_kernel_generation_oversample_pillbox_small kernel_func/kernel_func_OversampledPillboxSmall.cpp)
target_link_libraries(test_kernel_generation_oversample_pillbox_small ${TEST_LINK_LIBRARIES})

# Regular Sampling Pillbox
add_executable(test_kernel_generation_regular_sampling_pillbox kernel_func/kernel_func_RegularSamplingPillbox.cpp)
target_link_libraries(test_kernel_generation_regular_sampling_pillbox ${TEST_LINK_LIBRARIES})

# Regular Sampling Triangle
add_executable(test_kernel_generation_regular_sampling_triangle kernel_func/kernel_func_RegularSamplingTriangle.cpp)
target_link_libraries(test_kernel_generation_regular_sampling_triangle ${TEST_LINK_LIBRARIES})


# Gridder Functions Tests

# Single Pixel Overlap Pillbox
add_executable(test_gridder_bouns_checking gridder_func/gridder_func_BoundsChecking.cpp)
target_link_libraries(test_gridder_bouns_checking ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_multi_pixel_pillbox gridder_func/gridder_func_MultiPixelPillbox.cpp)
target_link_libraries(test_gridder_multi_pixel_pillbox ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_multiple_complex_vis gridder_func/gridder_func_MultipleComplexVis.cpp)
target_link_libraries(test_gridder_multiple_complex_vis ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_nearby_complex_vis gridder_func/gridder_func_NearbyComplexVis.cpp)
target_link_libraries(test_gridder_nearby_complex_vis ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_single_pixel_overlap_pillbox gridder_func/gridder_func_SinglePixelOverlapPillbox.cpp)
target_link_libraries(test_gridder_single_pixel_overlap_pillbox ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_small_pillbox gridder_func/gridder_func_SmallPillbox.cpp)
target_link_libraries(test_gridder_small_pillbox ${TEST_LINK_LIBRARIES})

# Triangle
add_executable(test_gridder_triangle gridder_func/gridder_func_Triangle.cpp)
target_link_libraries(test_gridder_triangle ${TEST_LINK_LIBRARIES})

# FractionalCoordToOversampledIndexMath
add_executable(test_gridder_fractional_coord_to_oversampled_index gridder_func/gridder_func_FractionalCoordToOversampledIndexMath.cpp)
target_link_libraries(test_gridder_fractional_coord_to_oversampled_index ${TEST_LINK_LIBRARIES})

# SteppedVsExactConvolution
add_executable(test_gridder_stepped_vs_exact_convolution gridder_func/gridder_func_SteppedVsExactConvolution.cpp)
target_link_libraries(test_gridder_stepped_vs_exact_convolution ${TEST_LINK_LIBRARIES})

# After the tests are build, declare them.
# Each unit test has its own executable for practicality.

# Convolution Tests
add_test(NAME ConvTriangleFunc COMMAND test_conv_triangle_func)
add_test(NAME ConvTopHatFunc COMMAND test_conv_tophat)
add_test(NAME ConvSincFunc COMMAND test_conv_sinc)
add_test(NAME ConvGaussianFunc COMMAND test_conv_gaussian)
add_test(NAME ConvGaussianSincFunc COMMAND test_conv_gaussian_sinc)

# Kernel Tests
add_test(NAME KernelGenerationOversampledPillbox COMMAND test_kernel_generation_oversampled_pillbox)
add_test(NAME KernelGenerationOversamplePillboxSmall COMMAND test_kernel_generation_oversample_pillbox_small)
add_test(NAME KernelGenerationRegularSamplingPillbox COMMAND test_kernel_generation_regular_sampling_pillbox)
add_test(NAME KernelGenerationRegularSamplingTriangle COMMAND test_kernel_generation_regular_sampling_triangle)

#Gridder Tests
add_test(NAME GridderBoundsChecking COMMAND test_gridder_bouns_checking)
add_test(NAME GridderMultiPixelPillbox COMMAND test_gridder_multi_pixel_pillbox)
add_test(NAME GridderMultipleComplexVis COMMAND test_gridder_multiple_complex_vis)
add_test(NAME GridderNearbyComplexVis COMMAND test_gridder_nearby_complex_vis)
add_test(NAME GridderSinglePixelOverlapPillbox COMMAND test_gridder_single_pixel_overlap_pillbox)
add_test(NAME GridderSmallPillbox COMMAND test_gridder_small_pillbox)
add_test(NAME GridderTriangle COMMAND test_gridder_triangle)
add_test(NAME GridderFractionalCoordToOversampledIndexMath COMMAND test_gridder_fractional_coord_to_oversampled_index)
add_test(NAME GridderStpeppedVsExactconvolution COMMAND test_gridder_stepped_vs_exact_convolution)

