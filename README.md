# Slow Transients Pipeline Prototype
## Project organisation
- doc: documentation files
  - html: auto-generated doxygen documentation
- src: source code files (.cpp and .h)
  - stp: the STP library code
  - stp-python: python bindings for STP library
  - reduce: command-line tools for the execution of the STP on arbitrary numpy data
  - test: unit tests for the STP library
  - benchmark: functions for benchmarking of STP library
  - auxiliary: external functions to auxiliate tests, benchmark and reduce module
  - third-party: external code, mostly libraries
- configs: auxiliary configuration files
- test-data: input test data files
- scripts: auxiliary scripts to generate FFTW plans and plot benchmarking results
- vagrant: virtual machine configuration

## Build & Run
### Dependencies

#### In Source (third-party)
- [Armadillo](http://arma.sourceforge.net/) [7.950.1]
- [Google Test](https://github.com/google/googletest) [1.8.0]
- [Google Benchmark](https://github.com/google/benchmark) [1.1.0]
- [cnpy](https://github.com/rogersce/cnpy) [repository head]
- [FFTW3](http://www.fftw.org/) [3.3.5]
- [pybind11](https://github.com/pybind/pybind11) [2.0.0]
- [TBB](https://www.threadingbuildingblocks.org/) [2017 Update 3]
- [OpenBLAS](http://www.openblas.net) [0.2.19]
- [RapidJSON](https://github.com/miloyip/rapidjson) [1.1.0]
- [TCLAP](http://tclap.sourceforge.net/) [1.2.1]
- [spdlog](https://github.com/gabime/spdlog) [0.14.0]
- [eigen](http://eigen.tuxfamily.org/) [3.3.4]
- [ceres](http://ceres-solver.org/) [1.13.0rc1]

### Clone

```sh
$ git clone https://github.com/SKA-ScienceDataProcessor/FastImaging.git
$ cd <path/to/project>
$ git submodule init
$ git submodule update
```

### Build

STP prototype is compiled using CMake tools. 
The following cmake options are available:

OPTION        | Description
------------- | -------------
 BUILD_TESTS           | Builds the unit tests (default=ON)
 BUILD_BENCHMARK       | Builds the benchmark tests (default=ON)
 USE_GLIBCXX_PARALLEL  | Uses GLIBCXX parallel mode (default=ON)
 USE_FLOAT             | Builds STP using FLOAT type to represent large structures of real/complex numbers (default=ON)
 WITH_FUNCTION_TIMINGS | Measures function execution times from the reduce executable (default=ON)
 USE_SERIAL_GRIDDER    | Uses serial implementation of gridder (default=OFF)
 USE_FFTSHIFT          | Performs shift of the image and beam matrices after fft (default=OFF)

The USE_FLOAT option is important, as it may affect the STP algorithm accuracy.
When compiled with USE_FLOAT=ON, most algorithm data structures of real or complex numbers will use single-precision floating-point type instead of double-precision floating-point type. 
In most systems the FLOAT type uses 4 bytes while DOUBLE uses 8 bytes. Thus, using FLOAT allows to reduce the memory usage and consequently the pipeline running time.

After building STP the FFTW plans shall be generated using the fftw-wisdom tool.
By using pre-generated FFTW plans the FFT step executes faster and the total running time of STP is smaller.
A script to generate FFTW plans using complex-to-real (c2r) FFT is provided in "project-root/scripts/fftw-wisdom" directory. 
The following matrix sizes are considered for FFTW plan generation:
128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536

When running the program only the above matrix sizes can be used as input image size, since the plans were generated only for these sizes. 
The full or relative pathname and filename of the FFTW plans for the complex-to-real (c2r) FFT step shall be given in the JSON configuration file.

#### Using a build script (Includes tests execution)
```sh
$ cd <path/to/project>
$ chmod +x build.sh
$ ./build.sh <OPTIONS>
```
OPTION | Description
-------| --------------
 -d    | Set CMAKE_BUILD_TYPE=Debug (default)
 -r    | Set CMAKE_BUILD_TYPE=Release
 -i    | Set CMAKE_BUILD_TYPE=RelWithDebInfo
 -f    | Set USE_FLOAT=ON (default is USE_FLOAT=OFF)
 -s    | Set USE_FFTSHIFT=ON (default is USE_FFTSHIFT=OFF)                                                                                                                                     
 -n    | Disable generation of fftw plans (enabled by default) 

#### Manually
```sh
$ mkdir -p <path/to/build/directory>
$ cd <path/to/build/directory>
$ cmake -DCMAKE_BUILD_TYPE=Release -DUSE_FLOAT=OFF <path/to/project/src>
$ make all -j4
```
Also, generate FFTW plans using fftw-wisdom (see OPTIONS of generate_wisdom.sh scritpt using --help):
```sh
$ cd <path/to/project>/scripts/fftw-wisdom
$ ./generate_wisdom.sh <OPTIONS>  
```
For instance, when compiled in Release mode with -DUSE_FLOAT=ON run:
```sh
$ ./generate_wisdom.sh -r -f
```

## Tests Execution
### Using CMake (after successful build)
```sh
$ cd path/to/build/directory
$ make test
```

### Running on the Command Line (all tests)
```sh
$ cd path/to/build/directory
$ run-parts ./tests
```

## Benchmark Execution
```sh
$ cd path/to/build/directory
$ make benchmarking
```

## STP Execution using Reduce module
The reduce executable is located in build-directory/reduce. It accepts the following arguments:

Argument | Usage  | Description
---------| -------| -------------
<input-file-json>  | required | Input JSON filename with configuration parameters (e.g. projectroot/configs/reduce/fastimg_oversampling_config.json)
<input-file-npz>   | required | Input NPZ filename with simulation data: uvw_lambda, vis, skymodel (e.g. projectroot/test-data/pipeline-tests/simdata_nstep10.npz) 
<output-file-json> | required | Output JSON filename for detected islands.
<output-file-npz>  | optional | Output NPZ filename for label map matrix (label_map).
-d, --diff | optional | Use residual visibilities - difference between 'input_vis' and 'model' visibilities.
-l, --log  | optional | Enable logger.

Example:
```sh
$ ./reduce projectroot/configs/reduce/fastimg_oversampling_config.json projectroot/test-data/pipeline-tests/simdata_nstep10.npz detected_islands.json -d -l
```
Note that the provided fastimg_oversampling_config.json file assumes that the FFTW wisdom files (with the pre-generated plans) are located in the current directory.
Thus, the path of the wisdom files in the JSON configuration file shall be properly setup. Otherwise, the wisdom files shall be copied from the projectroot/scripts/fftw-wisdom/wisdomfiles (or wisdomfiles_f when USE_FLOAT=ON) to the reduce directory.

## Code profiling
 - Valgrind framework tools can be used to profile STP library: callgrind (function call history and instruction profiling), cachegrind (cache and branch prediction profiling) and massif (memory profiling).
 - For more accurate profiling STP and reduce shall be compiled in Release mode. However, if detailed profiling of source code lines is desired, debug information shall be added (-g option of gcc). This can be done using the "Release With Debug Information" mode (CMAKE_BUILD_TYPE=RelWithDebInfo), as it uses compiling optimizations while adding debug symbols.
 - When running valgrind with callgrind tool, add --separate-threads=yes in order to independently profile all threads.
 - Results of these tools are written out to a file (default name: \<tool\>.out.\<pid\>).
 - Callgrind and cachegrind output files can be analyzed using kcachegrind GUI application, while massif output file can be analyzed with massif-visualizer.
 - Callgrind usage example:
```sh
$ mkdir -p path/to/build/directory
$ cd path/to/build/directory
$ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo path/to/project/src
$ make
$ cd reduce
$ valgrind --tool=callgrind --separate-threads=yes ./reduce -d projectroot/configs/reduce/fastimg_oversampling_config.json projectroot/test-data/pipeline-tests/simdata_nstep10.npz detected_islands.json -d -l
$ kcachegrind callgrind.out.*
```

## Release Notes
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
- Added gnuplot scripts to generate benchmark speedup plots

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
