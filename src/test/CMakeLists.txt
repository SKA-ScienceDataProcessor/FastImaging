# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Unit Tests

set(CMAKE_CXX_STANDARD 14)

# Dependency: Threads (required by google test)
find_package(Threads REQUIRED)

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include/rapidjson")

# For convenience all link libraries required by
# the tests are put in one variable
set(TEST_LINK_LIBRARIES stp cnpy-static gtest gtest_main Threads::Threads)

# Copy configuration files to the binary directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../config/imager-tests/conf_tests.json ${CMAKE_CURRENT_BINARY_DIR}/conf_tests.json COPYONLY)

# Configures the location of the test data
get_filename_component(IMAGER_DATAPATH "../../test-data/imager-data" ABSOLUTE)
add_definitions ( -D_IMAGER_TESTPATH="${IMAGER_DATAPATH}/")
get_filename_component(PIPELINE_DATAPATH "../../test-data/pipeline-data" ABSOLUTE)
add_definitions ( -D_PIPELINE_DATAPATH="${PIPELINE_DATAPATH}/")

# Configures the location of the configuration files
get_filename_component(PIPELINE_CONFIGPATH "../../config/pipeline-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_CONFIGPATH="${PIPELINE_CONFIGPATH}/")

# Configures the location of the cnpy test data
get_filename_component(CNPY_DATA_PATH "../../test-data/cnpy-data" ABSOLUTE)
add_definitions ( -D_CNPYTESTPATH="${CNPY_DATA_PATH}/")

# Helper function for test target specification
function(add_unit_test unit_test_name unit_test_source)
    add_executable(${unit_test_name} ${unit_test_source} ../auxiliary/fixtures.h ../auxiliary/gaussian2d.cpp
        ../auxiliary/load_data.cpp ../auxiliary/load_json_config.cpp imager/load_json_imager.cpp)
    add_dependencies(${unit_test_name} stp cnpy-static gtest gtest_main benchmark)
    target_include_directories(${unit_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../auxiliary
        ${RAPIDJSON_INCLUDE_PATH})
    target_link_libraries(${unit_test_name} ${TEST_LINK_LIBRARIES})
endfunction()

# Test Cases: Vector Math Functions ------------------------------------------------------------------------------------

add_unit_test(test_vector_math_funcs vectormath/test_vector_math_funcs.cpp)

# Test Cases: Convolution Functions ------------------------------------------------------------------------------------

# Triangle
add_unit_test(test_conv_triangle conv/conv_test_triangle.cpp)

# Tophat
add_unit_test(test_conv_tophat conv/conv_test_tophat.cpp)

# Sinc
add_unit_test(test_conv_sinc conv/conv_test_sinc.cpp)

# Gaussian
add_unit_test(test_conv_gaussian conv/conv_test_gaussian.cpp)

# Gaussian Sinc
add_unit_test(test_conv_gaussian_sinc conv/conv_test_gaussiansinc.cpp)

# Test Cases: Kernel Functions -----------------------------------------------------------------------------------------

# Oversampled Pillbox
add_unit_test(test_kernel_generation_oversampled_pillbox kernel/kernel_test_OversampledPillbox.cpp)

# Oversampled Pillbox Small
add_unit_test(test_kernel_generation_oversampled_pillbox_small kernel/kernel_test_OversampledPillboxSmall.cpp)

# Regular Sampling Pillbox
add_unit_test(test_kernel_generation_regular_sampling_pillbox kernel/kernel_test_RegularSamplingPillbox.cpp)

# Regular Sampling Triangle
add_unit_test(test_kernel_generation_regular_sampling_triangle kernel/kernel_test_RegularSamplingTriangle.cpp)

# Test Cases: Gridder Functions ----------------------------------------------------------------------------------------

# Bounds Checking
add_unit_test(test_gridder_bounds_checking gridder/gridder_test_BoundsChecking.cpp)

# Multi Pixel Pillbox
add_unit_test(test_gridder_multi_pixel_pillbox gridder/gridder_test_MultiPixelPillbox.cpp)

# Multiple Complex Vis
add_unit_test(test_gridder_multiple_complex_vis gridder/gridder_test_MultipleComplexVis.cpp)

# Nearby Complex Vis
add_unit_test(test_gridder_nearby_complex_vis gridder/gridder_test_NearbyComplexVis.cpp)

# Single Pixel Overlap Pillbox
add_unit_test(test_gridder_single_pixel_overlap_pillbox gridder/gridder_test_SinglePixelOverlapPillbox.cpp)

# Small Pixel
add_unit_test(test_gridder_small_pillbox gridder/gridder_test_SmallPillbox.cpp)

# Triangle
add_unit_test(test_gridder_triangle gridder/gridder_test_Triangle.cpp)

# FractionalCoordToOversampledIndexMath
add_unit_test(test_gridder_fractional_coord_to_oversampled_index gridder/gridder_test_FractionalCoordToOversampledIndexMath.cpp)

# FractionalCoordIn2DCase
add_unit_test(test_gridder_fractional_coord_in_2d_case gridder/gridder_test_FractionalCoordIn2DCase.cpp)

# KernelCaching
add_unit_test(test_gridder_kernel_caching gridder/gridder_test_KernelCaching.cpp)

# Oversampled Gridding
add_unit_test(test_gridder_oversampled_gridding gridder/gridder_test_OversampledGridding.cpp)

# Test Cases: Imager Functions -----------------------------------------------------------------------------------------

# Triangle
add_unit_test(test_imager_triangle imager/imager_test_Triangle.cpp)

# TopHat
add_unit_test(test_imager_tophat imager/imager_test_TopHat.cpp)

# Sinc
add_unit_test(test_imager_sinc imager/imager_test_Sinc.cpp)

# Gaussian
add_unit_test(test_imager_gaussian imager/imager_test_Gaussian.cpp)

# GaussianSinc
add_unit_test(test_imager_gaussiansinc imager/imager_test_GaussianSinc.cpp)

# Test Cases: Cnpy Functions -------------------------------------------------------------------------------------------

# Complex
add_unit_test(test_cnpy_load_complex cnpy/cnpy_test_complex.cpp)

# Float
add_unit_test(test_cnpy_load_float cnpy/cnpy_test_float.cpp)

# Test Cases: Fixtures Functions ---------------------------------------------------------------------------------------

add_unit_test(test_fixtures_model_generation_evaluation fixtures/fixtures_test_ModelGenerationEvaluation.cpp)

# Test Cases: SourceFind Functions -------------------------------------------------------------------------------------

# BasicSourceDetection
add_unit_test(test_sourcefind_basic_source_detection sourcefind/sourcefind_test_BasicSourceDetection.cpp)

# NegativeSourceDetection
add_unit_test(test_sourcefind_negative_source_detection sourcefind/sourcefind_test_NegativeSourceDetection.cpp)

# RmsEstimation
add_unit_test(test_sourcefind_rms_estimation sourcefind/sourcefind_test_RmsEstimation.cpp)

# Test Cases: Pipeline Functions ---------------------------------------------------------------------------------------
add_unit_test(test_pipeline_gaussiansinc pipeline/pipeline_test_gaussiansinc.cpp)

# Pass the test cases to CTest
# Note: for practicality each unit test has its own executable.

#Vector math
add_test(NAME VectorMathFunctions COMMAND test_vector_math_funcs)

# Convolution
add_test(NAME ConvTriangleFunc COMMAND test_conv_triangle)
add_test(NAME ConvTopHatFunc COMMAND test_conv_tophat)
add_test(NAME ConvSincFunc COMMAND test_conv_sinc)
add_test(NAME ConvGaussianFunc COMMAND test_conv_gaussian)
add_test(NAME ConvGaussianSincFunc COMMAND test_conv_gaussian_sinc)

# Kernel
add_test(NAME KernelGenerationOversampledPillbox COMMAND test_kernel_generation_oversampled_pillbox)
add_test(NAME KernelGenerationOversampledPillboxSmall COMMAND test_kernel_generation_oversampled_pillbox_small)
add_test(NAME KernelGenerationRegularSamplingPillbox COMMAND test_kernel_generation_regular_sampling_pillbox)
add_test(NAME KernelGenerationRegularSamplingTriangle COMMAND test_kernel_generation_regular_sampling_triangle)

# Gridder
add_test(NAME GridderBoundsChecking COMMAND test_gridder_bounds_checking)
add_test(NAME GridderMultiPixelPillbox COMMAND test_gridder_multi_pixel_pillbox)
add_test(NAME GridderMultipleComplexVis COMMAND test_gridder_multiple_complex_vis)
add_test(NAME GridderNearbyComplexVis COMMAND test_gridder_nearby_complex_vis)
add_test(NAME GridderSinglePixelOverlapPillbox COMMAND test_gridder_single_pixel_overlap_pillbox)
add_test(NAME GridderSmallPillbox COMMAND test_gridder_small_pillbox)
add_test(NAME GridderTriangle COMMAND test_gridder_triangle)
add_test(NAME GridderFractionalCoordToOversampledIndexMath COMMAND test_gridder_fractional_coord_to_oversampled_index)
add_test(NAME GridderFractionalCoordIn2DCase COMMAND test_gridder_fractional_coord_in_2d_case)
add_test(NAME GridderKernelCaching COMMAND test_gridder_kernel_caching)
add_test(NAME GridderOversampledGridding COMMAND test_gridder_oversampled_gridding)

# Imager
add_test(NAME ImagerTopHat COMMAND test_imager_tophat)
add_test(NAME ImagerTriangle COMMAND test_imager_triangle)
add_test(NAME ImagerSinc COMMAND test_imager_sinc)
add_test(NAME ImagerGaussian COMMAND test_imager_gaussian)
add_test(NAME ImagerGaussianSinc COMMAND test_imager_gaussiansinc)

# Cnpy
add_test(NAME CnpyLoadComplex COMMAND test_cnpy_load_complex)
add_test(NAME CnpyLoadFloat COMMAND test_cnpy_load_float)

# Fixtures
add_test(NAME FixturesModelGenerationEvaluation COMMAND test_fixtures_model_generation_evaluation)

# SourceFind
add_test(NAME SourceFindBasicSourceDetection COMMAND test_sourcefind_basic_source_detection)
add_test(NAME SourceFindNegativeSourceDetection COMMAND test_sourcefind_negative_source_detection)
add_test(NAME SourceFindRmsEstimation COMMAND test_sourcefind_rms_estimation)

# Pipeline Functions
add_test(NAME PipelineGaussianSincFunc COMMAND test_pipeline_gaussiansinc)
