# --- SKA Science Data Processor ---
# Slow Transients Pipeline (STP) Library Unit Tests

# Minimum language standard is C++17
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# Turn on useful compiler flags
add_compile_options(-march=native)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-ffast-math -O3)
endif()

# Dependency: Threads (required by google test)
find_package(Threads REQUIRED)

# Dependency: RapidJSON
set(RAPIDJSON_INCLUDE_PATH "../third-party/rapidjson/include")

# For convenience all link libraries required by
# the tests are put in one variable
set(TEST_LINK_LIBRARIES auxlib stp gtest gtest_main Threads::Threads armadillo)

# Copy configuration files to the binary directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../configs/imager-tests/conf_tests.json ${CMAKE_CURRENT_BINARY_DIR}/conf_tests.json COPYONLY)

# Configures the location of the test data
get_filename_component(CNPY_DATA_PATH "../../test-data/cnpy-tests" ABSOLUTE)
add_definitions ( -D_CNPYTESTPATH="${CNPY_DATA_PATH}/")
get_filename_component(IMAGER_DATAPATH "../../test-data/imager-tests" ABSOLUTE)
add_definitions ( -D_IMAGER_TESTPATH="${IMAGER_DATAPATH}/")
get_filename_component(VISIBILITY_DATAPATH "../../test-data/visibility-tests" ABSOLUTE)
add_definitions ( -D_VISIBILITY_TESTPATH="${VISIBILITY_DATAPATH}/")
get_filename_component(PIPELINE_DATAPATH "../../test-data/pipeline-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_DATAPATH="${PIPELINE_DATAPATH}/")

# Configures the location of the pipeline config files
get_filename_component(PIPELINE_CONFIGPATH "../../configs/pipeline-tests" ABSOLUTE)
add_definitions ( -D_PIPELINE_CONFIGPATH="${PIPELINE_CONFIGPATH}/")

# Helper function for test target specification
function(add_unit_test unit_test_name unit_test_source)
    add_executable(${unit_test_name} ${unit_test_source} imager/load_json_imager.cpp)
    add_dependencies(${unit_test_name} auxlib stp gtest gtest_main benchmark armadillo)
    target_include_directories(${unit_test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${RAPIDJSON_INCLUDE_PATH})
    target_link_libraries(${unit_test_name} ${TEST_LINK_LIBRARIES})
endfunction()

# Test Cases: Matrix Math Functions ------------------------------------------------------------------------------------

# Stats functions
add_unit_test(test_matrixmath_stat_funcs matrixmath/matrixmath_test_stat_funcs.cpp)

# Median functions
add_unit_test(test_matrixmath_median_funcs matrixmath/matrixmath_test_median_funcs.cpp)

# FFT shift
add_unit_test(test_matrixmath_fftshift matrixmath/matrixmath_test_fftshift.cpp)

# In-place matrix division
add_unit_test(test_matrixmath_inplacediv matrixmath/matrixmath_test_inplacediv.cpp)

# Matrix rotation
add_unit_test(test_matrixmath_rotate matrixmath/matrixmath_test_rotate.cpp)

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

# PSWF
add_unit_test(test_conv_pswf conv/conv_test_pswf.cpp)

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

# HalfPlane Shifted Gridding
add_unit_test(test_gridder_halfplane_shifted_gridding gridder/gridder_test_HalfPlaneShiftedGridding.cpp)

# Sample Weighting
add_unit_test(test_gridder_sample_weighting gridder/gridder_test_SampleWeighting.cpp)


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

# PSWF
add_unit_test(test_imager_pswf imager/imager_test_PSWF.cpp)


# Test Cases: Interpolation Functions --------------------------------------------------------------------------------------

add_unit_test(test_linear_interpolation interpolation/linear_interpolation_test.cpp)


# Test Cases: Visibility Functions -----------------------------------------------------------------------------------------

# Model visibilities
add_unit_test(test_visibility_model_generation visibility/visibility_test_ModelGeneration.cpp)

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

# Labeling
add_unit_test(test_sourcefind_labeling sourcefind/sourcefind_test_Labeling.cpp)

# Fitting
add_unit_test(test_sourcefind_fitting sourcefind/sourcefind_test_Fitting.cpp)

# Test Cases: Pipeline Functions ---------------------------------------------------------------------------------------
add_unit_test(test_pipeline_simple pipeline/pipeline_simple_test.cpp)
add_unit_test(test_pipeline_wprojection pipeline/pipeline_wprojection_test.cpp)


# Pass the test cases to CTest
# Note: for practicality each unit test has its own executable.

# Matrix math
add_test(NAME MatrixMathStatFuncs COMMAND test_matrixmath_stat_funcs)
add_test(NAME MatrixMathMedianFuncs COMMAND test_matrixmath_median_funcs)
add_test(NAME MatrixMathFFTShift COMMAND test_matrixmath_fftshift)
add_test(NAME MatrixMathInplaceDiv COMMAND test_matrixmath_inplacediv)
add_test(NAME MatrixMathRotate COMMAND test_matrixmath_rotate)

# Convolution
add_test(NAME ConvTriangleFunc COMMAND test_conv_triangle)
add_test(NAME ConvTopHatFunc COMMAND test_conv_tophat)
add_test(NAME ConvSincFunc COMMAND test_conv_sinc)
add_test(NAME ConvGaussianFunc COMMAND test_conv_gaussian)
add_test(NAME ConvGaussianSincFunc COMMAND test_conv_gaussian_sinc)
add_test(NAME ConvPswfFunc COMMAND test_conv_pswf)

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
add_test(NAME GridderHalfplaneShiftedGridding COMMAND test_gridder_halfplane_shifted_gridding)
add_test(NAME GridderSampleWeighting COMMAND test_gridder_sample_weighting)

# Imager
add_test(NAME ImagerTopHat COMMAND test_imager_tophat)
add_test(NAME ImagerTriangle COMMAND test_imager_triangle)
add_test(NAME ImagerSinc COMMAND test_imager_sinc)
add_test(NAME ImagerGaussian COMMAND test_imager_gaussian)
add_test(NAME ImagerGaussianSinc COMMAND test_imager_gaussiansinc)
add_test(NAME ImagerPSWF COMMAND test_imager_pswf)

# Interpolation
add_test(NAME LinearInterpolation COMMAND test_linear_interpolation)

# Visibility
add_test(NAME VisibilityModelGeneration COMMAND test_visibility_model_generation)

# Cnpy
add_test(NAME CnpyLoadComplex COMMAND test_cnpy_load_complex)
add_test(NAME CnpyLoadFloat COMMAND test_cnpy_load_float)

# Fixtures
add_test(NAME FixturesModelGenerationEvaluation COMMAND test_fixtures_model_generation_evaluation)

# SourceFind
add_test(NAME SourceFindBasicSourceDetection COMMAND test_sourcefind_basic_source_detection)
add_test(NAME SourceFindNegativeSourceDetection COMMAND test_sourcefind_negative_source_detection)
add_test(NAME SourceFindRmsEstimation COMMAND test_sourcefind_rms_estimation)
add_test(NAME SourceFindLabeling COMMAND test_sourcefind_labeling)
add_test(NAME SourceFindFitting COMMAND test_sourcefind_fitting)

# Pipeline Functions
add_test(NAME SlowTransientsPipelineSimple COMMAND test_pipeline_simple)
add_test(NAME SlowTransientsPipelineWprojection COMMAND test_pipeline_wprojection)

