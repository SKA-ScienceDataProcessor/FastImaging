## Release Notes
### 22 August 2018
- Fix build issue
- Build static library for auxiliary functions

### 20 August 2018
- Implemented support for W-projection which is used for wide-field imaging
- Implemented initial support for A-projection (work in progress)
- Performed several improvements and fixes
- Added ffast-math gcc option

### 9 February 2018
- Implemented 8-connected component labeling
- Added option to select between 4- and 8-connected labeling
- Added option to define minimum area of a source to be considered as valid
- Added functionality to fix image size when it is not multiple of 4
- Added new unit tests

### 30 September 2017
- Improved parallel scalability of labeling function
- Fixed computation of bounding box using multiple threads

### 29 September 2017
- Added Changelog file
- Updated usage description in Readme file
- Enable CMake target to generate inplace fftw wisdom files (to allow benchmarking of in-place FFT)

### 22 September 2017
- Added option to select median function
- Added option to disable negative source detection from reduce
- Moved conversion of half-plane visibilities to the gridder
- Added fftshift benchmark
- Updated benchmarks to use larger image sizes 

### 15 September 2017
- Implemented method of moments for initial gaussian fitting
- Changed output data for each island
- Removed compute_barycentre option
- Improved documentation

### 29 August 2017
- Improved doxygen documentation
- Added CMake target to generate FFTW wisdom files
- Fixed bugs

### 25 August 2017
- Added option to use analytic derivatives for gaussian fitting using ceres-solver
- Improved implementation of gaussian fitting
- Added unit test for gaussian fitting
- Added doxygen documentation 

### 4 August 2017
- Implemented gaussian fitting using ceres-solver
- Fixed bugs

### 28 July 2017
- Implemented baseline weighting feature
- Changed normalisation of fft output by using weighting information
- Made generation of beam signal an optional step
- Added CMake option to explicitly use fftshift function after the FFT step and perform labeling over shifted matrices
- Fixed issue on derivation of out-of-bound kernels that could result in a non-symmetric gridded matrix

### 14 July 2017
- Implemented alternative method to compute exact median based on the binmedian algorithm - it is faster than nth_element when run in multiple cores
- Improved/fixed unit tests and benchmarks

### 5 July 2017
- Implemented generation of model visibilities from input skymodel and UVW baselines
- Updated test data 
- Improved/fixed python-bindings
- Improved/fixed unit tests and benchmarks

### 28 June 2017
- Implemented HalfComplex-to-Real FFT
- Removed matrix shifting steps before and after FFT
- Modified Connected Components Labeling algorithm to process not shifted matrix
- Added new matrix class that uses calloc method
- Implemented parallel algorithm for Connected Components Labeling
- Rewrote some parts of source find step to minimize memory accesses (e.g. merged some functions)
- Improved RMS estimation using incremental standard deviation
- When possible reused computed values of mean, sigma and median among several functions
- Implemented binapprox method to estimate an approximation of median (it is faster)
- Added several improvements (e.g. compare floats using integer operations in RMS estimation function)
- Fixed FFTW to work with 2¹⁶ x 2¹⁶ matrices

### 22 May 2017
- Added support to FFTW Wisdom files
- Added OpenBlas to in-source third-party libraries
- Enabled sse, avx, avx2 configure options for FFTW
- Implemented real-to-complex FFT for the beam matrix
- Replaced backward FFT by forward FFT by applying the FFT duality property
- Added new options to JSON file
- Added cmake option to use float rather than double type to represent large structures of real/complex numbers
- Implemented shifted-gridder which generates shifted matrices
- Implemented MatStp class which creates a matrix initialized with zeros using calloc (inherits arma::Mat<>)
- Updated data matrices to use shorter primitive types
- Added and improved benchmark tests
- Performed general improvements to STP implementation

### 7 March 2017
- Some minor improvements and fixes
- Improved benchmark tests
- Added new benchmark tests
- Improve gridder for the case of exact kernel

### 8 February 2017
- Some minor improvements
- Added RapidJSON, TCLAP and spdlog in-source
- Improved benchmarks

### 7 February 2017
- Fixed minor bugs

### 3 February 2017
- Added armadillo to third-party libraries
- Added support for GNU parallel mode of libstdc++ (useful for nth_element function used by arma::median)
- Implemented new benchmark functions
- Added new option in reduce to enable/disable usage of residual visibilities
- Reduced maximum peak of memory usage (based on memory profiling)
- Improved STP performance, code style and comments 

### 27 January 2017
- Added new python binding over source_find_image function
- Improved existing python binding over image_visibilities function
- Removed std::optional for rms_est 
- Implemented new unitary tests

### 26 January 2017
- Improved matrix shift implementation
- Re-implemented reduce target
- Implemented new run_imagevis and run_sourcefind targets

### 24 January 2017
- Removed visibility and pipeline header files
- Created new tests for pipeline
- Created new benchmarks for source-find, imager, fft
- Moved JSON functions to auxiliary directory
- Added new simulation data for benchmarking and tests
- Removed WITH_ARMAFFT cmake option
- Fixed source find bug: computed bg_level based on median
- Moved fixtures and gaussian 2D to auxiliary folder
- Removed data image from island parameters
- Moved searching of the extremum value index from island constructor to label dectection functions
- Added new vector math functions with TBB parallelization
- Improved implementation of estimate RMS using TBB parallelization
- Improved implementation of islands constructor
- Added separate functions for finding positive and negative sources
- Added several improvements and fixes, mostly for gridder and source find
- Parallelized accumulate, mean, stddev and shift operations
- Disabled Armadillo runtime linking library (now links directly to OpenBLAS)
- Added OpenBLAS function to perform inplace division of a matrix by a constant
- Parallelized several functions of source find and image visibilities
- Added config folder
- Added test-data as git sub-module

### 17 January 2017
- Created three branchs:
  - master: optimized multithreaded version
  - reference-optimized: optimized single-threaded version
  - reference-unopt: non-optimized single-threaded version
- Fixed bug in kernel bound checking code
- Added new kernel_exact parameter to disable oversampling
- Added new JSON parameters for reduce
- Implemented new benchmarks: populate_kernel_cache, gridder and pipeline
- Added TBB library to third-party
- Parallelized gridder (oversampling case) using TBB
- Enabled multithreaded FFTW
- Disabled armadillo DEBUG flag 

### 5 January 2017
- Renamed libstp to stp
- Added namespace stp
- Removed STP-Runner
- Added Reduce module for simulated run of pipeline
- Implemented visibity module
- Implemented python bindings for the "image_visibilities" function, as per imager.py (fastimgproto/bindings)
- Renamed pipeline.h to imager.h
- Implemented new pipeline module based on simpipe.py (fastimgproto/scripts)
- Implemented sigma_clip function
- Added std::optional to represent oversampling and rms
- Added new tests and improved existing ones
- Defined benchmark functions in new folder (not implemented yet)
- Fixed reading of fortran_order flag in cnpy functions 
- Applied several optimizations to stp functions

### 9 December 2016
- Implemented 1st version of the SourceFindImage (IslandParams and SourceFindImage structs/functions)
- Implemented 1st version of the Fixtures (functions uncorrelated_gaussian_noise_background and evaluate_model_on_pixel_grid)
- Implemented Connected Components Labeling function 
- Implemented Gaussian2D class (based on Gaussian2D of the astropy library)
- Created test environment for: SourceFind and Fixtures modules
- Added Google Benchmark functions for all tests except 1D convolution
- Rename some files/variables/functions/structs 

### 23 November 2016
- FFTW3 is now an in-source dependency
- Added rapidjson to the source dependencies
- Implemented 1st version of the pipeline (function: image_visibilities)
- Added tests for pipeline and cnpy
- Completely changed the implementation of the STP Runner
- Added JSON files with configuration values for the STP Runner and tests
- Reduced tolerance value for all tests

### 9 November 2016
- Google Test is now an in-source dependency
- Added Google Benchmark to the in-source dependencies directory
- CMake scripts were greatly improved
- Added convenience bash script for building the source tree
- Code fixes to the gridder/convolution functions
- Code fixes to the STP Runner

### 3 November 2016
- Implemented 2nd version of the gridder (functions: populate_kernel_cache and calculate_oversampled_kernel_indices)
- Created test environment for: Triangle convolution [on gridder], StpeppedVsExactconvolution and FractionalCoordToOversampledIndexMath

### 28 October 2016
- Updated 1D kernel functions: using template classes
- Updated 2D kernel function: use of templates and functors
- Implemented 1st version of the gridder (functions: bounds_check_kernel_centre_locations and convolve_to_grid)
- Created test environment for the Tophat convolution [on gridder]
- Updated test environment for the STP library
- [Removed "using namespaces"](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/8)

### 19 October 2016
- Revised project structure
- Created first version of the STP-Runner
- Task from _Development Plan_
  - [Integrate cnpy into build, convert to/from Armadillo arrays, test.](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/2)

### 14 October 2016
- Revised project structure
- Tasks from _Development Plan_
  - [Basic 1D convolution functions(tophat, triangle, sinc, triangle and gaussiansinc)](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/3)
  - [2-d kernel generation from convolution functions (tophat and triangle)](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/6)
- Created a test environment for the STP library functions so far
