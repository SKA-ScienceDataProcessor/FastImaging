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

# Dependency: rapidjson
find_path(RAPIDJSON rapidjson)
if(${RAPIDJSON} STREQUAL "RAPIDJSON-NOTFOUND")
	message(FATAL_ERROR "rapidjson not found")
endif()

# TODO: Dependency: OpenBLAS
# find_package(BLAS REQUIRED)

# Dependency: auxiliary
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/../../auxiliary)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_BINARY_DIR}/../../auxiliary)
endif()


# For convenience all link libraries required by
# the tests are put in one variable
set(TEST_LINK_LIBRARIES ${STP_TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main benchmark)

# Copy configuration files to the binary directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/pipeline_data/conf_tests.json ${CMAKE_CURRENT_BINARY_DIR}/conf_tests.json COPYONLY)

# Configures the pipeline_func.h on the location of the test data
get_filename_component(PIPELINE_PATH "data/pipeline_data/" ABSOLUTE)
add_definitions ( -D_TESTPATH="${PIPELINE_PATH}/")
configure_file(pipeline_func/pipeline_func.h pipeline_func/pipeline_func.h @ONLY)


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

# Bounds Checking
add_executable(test_gridder_bouns_checking gridder_func/gridder_func_BoundsChecking.cpp)
target_link_libraries(test_gridder_bouns_checking ${TEST_LINK_LIBRARIES})

# Multi Pixel Pillbox
add_executable(test_gridder_multi_pixel_pillbox gridder_func/gridder_func_MultiPixelPillbox.cpp)
target_link_libraries(test_gridder_multi_pixel_pillbox ${TEST_LINK_LIBRARIES})

# Multiple Complex Vis
add_executable(test_gridder_multiple_complex_vis gridder_func/gridder_func_MultipleComplexVis.cpp)
target_link_libraries(test_gridder_multiple_complex_vis ${TEST_LINK_LIBRARIES})

# Nearby Complex Vis
add_executable(test_gridder_nearby_complex_vis gridder_func/gridder_func_NearbyComplexVis.cpp)
target_link_libraries(test_gridder_nearby_complex_vis ${TEST_LINK_LIBRARIES})

# Single Pixel Overlap Pillbox
add_executable(test_gridder_single_pixel_overlap_pillbox gridder_func/gridder_func_SinglePixelOverlapPillbox.cpp)
target_link_libraries(test_gridder_single_pixel_overlap_pillbox ${TEST_LINK_LIBRARIES})

# Small Pixel
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


# Pipeline Functions Tests

# Triangle
add_executable(test_pipeline_triangle pipeline_func/pipeline_func_Triangle.cpp pipeline_func/pipeline_func.h)
target_include_directories(test_pipeline_triangle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(test_pipeline_triangle ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# TopHat
add_executable(test_pipeline_tophat pipeline_func/pipeline_func_TopHat.cpp pipeline_func/pipeline_func.h)
target_include_directories(test_pipeline_tophat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(test_pipeline_tophat ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# Sinc
add_executable(test_pipeline_sinc pipeline_func/pipeline_func_Sinc.cpp pipeline_func/pipeline_func.h)
target_include_directories(test_pipeline_sinc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(test_pipeline_sinc ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# Gaussian
add_executable(test_pipeline_gaussian pipeline_func/pipeline_func_Gaussian.cpp pipeline_func/pipeline_func.h)
target_include_directories(test_pipeline_gaussian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(test_pipeline_gaussian ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# GaussianSinc
add_executable(test_pipeline_gaussiansinc pipeline_func/pipeline_func_GaussianSinc.cpp pipeline_func/pipeline_func.h)
target_include_directories(test_pipeline_gaussiansinc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy)
target_link_libraries(test_pipeline_gaussiansinc ${TEST_LINK_LIBRARIES} cnpy auxiliary)


# Cnpy Function Tests

# Complex
add_executable(test_cnpy_load_complex cnpy_func/cnpy_func_complex.cpp)
target_include_directories(test_cnpy_load_complex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary)
target_link_libraries(test_cnpy_load_complex ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# Float
add_executable(test_cnpy_load_float cnpy_func/cnpy_func_float.cpp)
target_include_directories(test_cnpy_load_float PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cnpy ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary)
target_link_libraries(test_cnpy_load_float ${TEST_LINK_LIBRARIES} cnpy auxiliary)

# After the tests are built, declare them.
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

# Gridder Tests
add_test(NAME GridderBoundsChecking COMMAND test_gridder_bouns_checking)
add_test(NAME GridderMultiPixelPillbox COMMAND test_gridder_multi_pixel_pillbox)
add_test(NAME GridderMultipleComplexVis COMMAND test_gridder_multiple_complex_vis)
add_test(NAME GridderNearbyComplexVis COMMAND test_gridder_nearby_complex_vis)
add_test(NAME GridderSinglePixelOverlapPillbox COMMAND test_gridder_single_pixel_overlap_pillbox)
add_test(NAME GridderSmallPillbox COMMAND test_gridder_small_pillbox)
add_test(NAME GridderTriangle COMMAND test_gridder_triangle)
add_test(NAME GridderFractionalCoordToOversampledIndexMath COMMAND test_gridder_fractional_coord_to_oversampled_index)
add_test(NAME GridderStpeppedVsExactconvolution COMMAND test_gridder_stepped_vs_exact_convolution)

# Pipeline Tests
add_test(NAME PipelineTopHat COMMAND test_pipeline_tophat)
add_test(NAME PipelineTriangle COMMAND test_pipeline_triangle)
add_test(NAME PipelineSinc COMMAND test_pipeline_sinc)
add_test(NAME PipelineGaussian COMMAND test_pipeline_gaussian)
add_test(NAME PipelineGaussianSinc COMMAND test_pipeline_gaussiansinc)

# Cnpy Tests
add_test(NAME CnpyLoadComplex COMMAND test_cnpy_load_complex)
add_test(NAME CnpyLoadFloat COMMAND test_cnpy_load_float)

