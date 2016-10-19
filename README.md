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
  - third-party: external code, such as libraries
- tools: auxiliary tools

## Build & Run
### Dependencies
- Armadillo [v 7.400.2]
- Google Test [v 1.7.0]
- Tclap [v 1.2.1]
- Spdlog [v 1.12]
- Cnpy [included in sources]

### Build
- on the project top-level directory create a "build" directory and cd into it
- mkdir build
- mkdir build/stp-runner
- cd build
- cmake ../src/libstp
- make
- cd stp-runner
- cmake ../../../src/stp-runner
- make

## Tests Execution
### Using CMake
- cd build
- make test

### Running on the Command Line (all tests)
- cd build
- run-parts ./bin/tests

## STP-Runner Execution
- cd build
- ./bin/stp-runner/stp-runner
- can receive up to four arguments:
   - REQUIRED mode (-m):
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

- *Example:* ./bin/stp-runner/stp-runner -m tophat -f mock_uvw_vis.npz -k vis -p

## Known Issues
- Gaussian and Gaussian-Sinc convolution kernels fails with one specific file made available by Tim. To be reviewed in next release.

## Release Notes
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
