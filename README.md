# Slow Transients Pipeline Prototype
## Filesystem organisation
- doc: documentation files
  - html: auto-generated documentation (e.g. doxygen, LaTeX)
- reference: reference code published on Confluence
- include: public header files
- src: source code files (.cpp and .h)
  - libstp: the STP library code
  - test: unit tests for the STP library
  - stp-runner: allows the execution of any STP function on arbitrary numpy data
  - third-party: external code, mostly libraries
- tools: auxiliary tools

## Build & Run
### Dependencies

#### External (Debian-provided)
- [Armadillo](http://arma.sourceforge.net/) [7.500.0]
- [TCLAP](http://tclap.sourceforge.net/) [1.2.1]
- [Spdlog](https://github.com/gabime/spdlog) [0.11.0]

#### In Source (third-party)
- [Google Test](https://github.com/google/googletest) [1.8.0]
- [Google Benchmark](https://github.com/google/benchmark) [1.1.0]
- [cnpy](https://github.com/rogersce/cnpy) [repository head]

### Build

#### Using a build script (Includes tests execution)
- cd to the project's top-level directory
- chmod +x build.sh
- ./build.sh

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

## STP-Runner Execution
- cd build folder
- ./stp-runner/stp-runner
- can receive up to four arguments:
   - REQUIRED action (-a):
      - tophat for tophat function
      - sinc for sinc function
      - gaussian for gaussian function
      - gaussian-sinc for gaussian-sinc function
      - triangle for triangle function
      - tophat-kernel to generate a tophat kernel
      - triangle-kernel to generate a triangle kernel
   - REQUIRED filepath (-f):
      - path/to/input/data: can be a npy or npz file
   - key (-k): 
      - key value to select data from npz file. If not present, stp-runner will assume a npy file
   - print (-p):
      - optional flag to print to an external file named "out.txt"

- *Example:* ./bin/stp-runner/stp-runner -a tophat -f mock_uvw_vis.npz -k vis -p

## Release Notes
### 9 November 2016
- Google Test is now an in-source dependency
- Added Google Benchmark to the in-source dependencies directory
- CMake scipts were greatly improved
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
