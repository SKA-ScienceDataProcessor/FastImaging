cmake_minimum_required(VERSION 2.8.9)
project (stp)

# Sets the convolution directory
set(CONV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/convolution/)
include_directories(${CONV_DIR})
set(CONV_FILES ${CONV_DIR}conv_func.h ${CONV_DIR}conv_func.cpp)
set(KERNEL_FILES ${CONV_DIR}kernel_func.h ${CONV_DIR}kernel_func.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../test ${CMAKE_CURRENT_BINARY_DIR}/test)

add_library(stp STATIC ${CONV_FILES} ${KERNEL_FILES} ${GRIDDER_FILES})

enable_testing()

add_test(NAME ConvTriangleFunc COMMAND test_conv_triangle_func)
add_test(NAME ConvTopHatFunc COMMAND test_conv_tophat)
add_test(NAME ConvSincFunc COMMAND test_conv_sinc)
add_test(NAME ConvGaussianFunc COMMAND test_conv_gaussian)
add_test(NAME ConvGaussianSincFunc COMMAND test_conv_gaussian_sinc)

add_test(NAME KernelGenerationOversampledPillbox COMMAND test_kernel_generation_oversampled_pillbox)
add_test(NAME KernelGenerationOversamplePillboxSmall COMMAND test_kernel_generation_oversample_pillbox_small)
add_test(NAME KernelGenerationRegularSamplingPillbox COMMAND test_kernel_generation_regular_sampling_pillbox)
add_test(NAME KernelGenerationRegularSamplingTriangle COMMAND test_kernel_generation_regular_sampling_triangle)
