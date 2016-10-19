# STP runner

## Build & Run
### Dependencies
- armadillo 7.400.2
- tclap 1.2.1
- spdlog 1.12
- cnpy

### Build
- go to build folder
- build the libstp
   - cmake ../src/libstp/
   - make
- mkdir stp-runner
- go to stp-runner folder
- commands:
    - cmake ../../../src/stp-runner
    - make

### Running
- go to the build directory in stp-runner and run ./stp-runner
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

## Release Notes
### 18 October 2016
- [Integrate cnpy into build, convert to/from Armadillo arrays, test.](https://github.com/SKA-ScienceDataProcessor/FastImaging/issues/2)
