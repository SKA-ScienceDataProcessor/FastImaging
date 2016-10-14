# Slow Transients Pipeline Prototype
## Filesystem organisation
- doc: documentation files
  - html: auto-generated documentation (e.g. doxygen, LaTeX)
- reference: reference code published on Confluence
- include: public header files
- src: source code files (.cpp and .h)
  - libstp: the STP library code
  - test: unit tests for the STP library
  - third-party: external code, such as libraries
- tools: auxiliary tools

## Build & Run
### Dependencies
- Armadillo [v 7.400.2]
- Google Test [v 1.7.0]

### Build
- on the project top-level directory create a "build" directory and cd into it
- mkdir build
- cd build
- cmake ../src/libstp
- make

## Tests execution
### Using CMake
- cd build
- make test

### Running on the Command Line (all tests)
- cd build
- run-parts ./bin

## Release Notes
### 14 October 2016
- Revised project structure
- Tasks from _Development Plan_
  - [Basic 1D convolution functions(tophat, triangle, sinc, triangle and gaussiansinc)](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/3)
  - [2-d kernel generation from convolution functions (tophat and triangle)](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/6)
- Created a test environment for the STP library functions so far
