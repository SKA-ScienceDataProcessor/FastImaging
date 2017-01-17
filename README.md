# Slow Transients Pipeline Prototype
## Project organisation
- doc: documentation files
  - html: auto-generated documentation (e.g. doxygen, LaTeX)
- reference: reference code published on Confluence
- src: source code files (.cpp and .h)
  - stp: the STP library code
  - stp-python: python bindings for STP library
  - reduce: command-line tool for the execution of the STP on arbitrary numpy data
  - test: unit tests for the STP library
  - benchmark: functions for benchmarking of STP library
  - auxiliary: external functions to auxiliate tests, benchmark and reduce module
  - third-party: external code, mostly libraries
- config: auxiliary configuration files
- test-data: input test data files
- plot-scripts: scripts to plot benchmarking results
- vagrant: virtual machine configuration

## Build & Run
### Dependencies

#### External (Debian-provided)
- [Armadillo](http://arma.sourceforge.net/) [7.500.2]
- [TCLAP](http://tclap.sourceforge.net/) [1.2.1]
- [Spdlog](https://github.com/gabime/spdlog) [0.11.0]
- [rapidjson](https://github.com/miloyip/rapidjson) [0.12]

#### In Source (third-party)
- [Google Test](https://github.com/google/googletest) [1.8.0]
- [Google Benchmark](https://github.com/google/benchmark) [1.1.0]
- [cnpy](https://github.com/rogersce/cnpy) [repository head]
- [FFTW3](http://www.fftw.org/) [3.3.5]
- [pybind11](https://github.com/pybind/pybind11) [2.0.0]

### Build

#### Using a build script (Includes tests execution)
- cd to the project's top-level directory
- chmod +x build.sh
- ./build.sh (add 'r' option to build in release mode)

#### Manually
- Create a build directory: mkdir -p path/to/build/directory
- cd path/to/build/directory
- cmake path/to/project/src
- make

## Tests Execution
### Using CMake (after successful build)
- cd path/to/build/directory
- make test

### Running on the Command Line (all tests)
- cd path/to/build/directory
- run-parts ./libstp/tests

## Benchmark Execution
- cd path/to/build/directory
- make benchmarking

## STP Execution using Reduce module
- cd build folder
- ./reduce/reduce
- receives two mandatory arguments:
   - REQUIRED input filename (-f):
      - path/to/input/data: must be a npz file
   - REQUIRED json configuration filename (-c):
      - path/to/input/config: must be a json file (see example: projectroot/config/fastimg_config.json)

- *Example:* ./bin/reduce/reduce -f ./mock_uvw_vis.npz -c ./fastimg_config.json

## Release Notes
### 17 January 2017
- Created three branchs:
  - master: optimized multithreaded version
  - reference-optimized: optimized single-threaded version
  - reference-unopt: non-optimized single-threaded version
- Fixed bug in kernel bound checking code
- Added new kernel_exact parameter to disable oversampling
- Significantly improved populate_kernel_cache function
- Added new JSON parameters for reduce
- Implementated new benchmarks: populate_kernel_cache, gridder and pipeline
- Added TBB library to third-party
- Parallelized gridder (oversampling case) using TBB
- Enabled multithreaded FFTW
- Disabled armadillo DEBUG flag 
- Added gnuplot scripts to generate benchmark speedup plots
- Performed several other improvements 

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
