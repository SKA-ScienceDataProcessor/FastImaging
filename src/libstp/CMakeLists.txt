cmake_minimum_required(VERSION 2.8.9)
project (stp)

#Sets to build in debug mode. Remove on release
set(CMAKE_BUILD_TYPE Debug)

# Sets the convolution directory
set(CONV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/convolution/)
include_directories(${CONV_DIR})

# Sets the gridder directory
set(GRIDDER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gridder/)
include_directories(${GRIDDER_DIR})

set(CONV_FILES ${CONV_DIR}conv_func.h ${CONV_DIR}conv_func.cpp)
set(GRIDDER_FILES ${GRIDDER_DIR}gridder.h ${GRIDDER_DIR}gridder.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../test ${CMAKE_CURRENT_BINARY_DIR}/test)

add_library(stp STATIC ${CONV_FILES} ${GRIDDER_FILES})

enable_testing()

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
