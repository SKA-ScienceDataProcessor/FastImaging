# Slow Transients Pipeline Prototype
## Project organisation
- doc: documentation files
  - html: auto-generated documentation (e.g. doxygen, LaTeX)
- reference: reference code published on Confluence
- src: source code files (.cpp and .h)
  - stp: the STP library code
  - stp-python: python bindings for STP library
  - reduce: command-line tools for the execution of the STP on arbitrary numpy data
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
- [OpenBLAS](http://www.openblas.net) [0.2.19]
- [LAPACK](http://www.netlib.org/lapack) [3.7.0]
- [rapidjson](https://github.com/miloyip/rapidjson) [0.12]
- [TCLAP](http://tclap.sourceforge.net/) [1.2.1]
- [Spdlog](https://github.com/gabime/spdlog) [0.11.0]

#### In Source (third-party)
- [Armadillo](http://arma.sourceforge.net/) [7.600.2]
- [Google Test](https://github.com/google/googletest) [1.8.0]
- [Google Benchmark](https://github.com/google/benchmark) [1.1.0]
- [cnpy](https://github.com/rogersce/cnpy) [repository head]
- [FFTW3](http://www.fftw.org/) [3.3.5]
- [pybind11](https://github.com/pybind/pybind11) [2.0.0]
- [TBB](https://www.threadingbuildingblocks.org/) [2017 Update 3]

### Build

#### Using a build script (Includes tests execution)
```sh
$ cd <project directory>
$ chmod +x build.sh
$ ./build.sh    #(add 'r' option to build in release mode)
```

#### Manually
```sh
$ mkdir -p path/to/build/directory
$ cd path/to/build/directory
$ cmake path/to/project/src
$ make
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
$ run-parts ./libstp/tests
```

## Benchmark Execution
```sh
$ cd path/to/build/directory
$ make benchmarking
```

## STP Execution using Reduce module
- cd into reduce folder located in path/to/build/directory
- run reduce binary using the following arguments:
   <input-file-json> : (required)  Input JSON filename with configuration parameters.
   <input-file-npz> : (required)  Input NPZ filename with simulation data (uvw_lambda, model, vis).
   <output-file-json> : (required)  Output JSON filename for detected islands.
   <output-file-npz> : (optional)  Output NPZ filename for label map matrix (label_map).
   -d,  --diff : (optional) Use residual visibilities - difference between 'input_vis' and 'model' visibilities.
   -l,  --log : (optional)  Enable logger.
- Example:
```sh
$ ./reduce projectroot/config/pipeline-benchmark/fastimg_oversampling_config.json projectroot/test-data/pipeline-data/simdata_small.npz detected_islands.json -l
```

## Code profiling
 - Valgrind framework and kcachegrind GUI application can be used to profile STP library. 
 - Recommended valgrind tools: callgrind (function call history and instruction profiling), cachegrind (cache and branch prediction profiling) and massif (memory profiling).
 - STP and reduce shall be compiled in Release mode for realistic profiling. However, if detailed profiling of source code is desired, debug info (-g option of gcc) shall be added. This can be accomplished "Release With Debug Information" mode (CMAKE_BUILD_TYPE=RelWithDebInfo) as this mode keeps compiling optimizations while adding debug symbols.
 - When running valgrind with callgrind tool, add --separate-threads=yes if you want to profile all threads.
 - Results of these tools are written out to a file (default name: <tool>.out.<pid>).
 - Callgrind and cachegrind output files can be analyzed using kcachegrind, while massif output file can be analyzed with massif-visualizer.
 - Callgrind usage example:
```sh
$ mkdir -p path/to/build/directory
$ cd path/to/build/directory
$ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo path/to/project/src
$ make
$ cd reduce
$ valgrind --tool=callgrind --separate-threads=yes ./reduce projectroot/config/pipeline-benchmark/fastimg_oversampling_config.json projectroot/test-data/pipeline-data/simdata_small.npz detected_islands.json
$ kcachegrind callgrind.out.*
```

## Release Notes
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
