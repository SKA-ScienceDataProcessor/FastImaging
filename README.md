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
- scripts: auxiliary scripts to generate FFTW wisdom files and test python bindings
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
- [Eigen](http://eigen.tuxfamily.org/) [3.3.4]
- [Ceres Solver](http://ceres-solver.org/) [1.13.0rc1]


### Clone

```sh
$ git clone https://github.com/SKA-ScienceDataProcessor/FastImaging.git
$ cd <path/to/project>
$ git submodule init
$ git submodule update    # Update test-data directory
```

### Build

To build the STP prototype, the following CMake options are available:

OPTION        | Description
------------- | -------------
BUILD_TESTS           | Builds the unit tests (default=ON)
BUILD_BENCHMARK       | Builds the benchmark tests (default=ON)
USE_GLIBCXX_PARALLEL  | Uses GLIBCXX parallel mode - required for parallel nth_element (default=ON)
USE_FLOAT             | Builds STP using FLOAT type to represent large arrays of real/complex numbers (default=ON)
WITH_FUNCTION_TIMINGS | Measures function execution times from the reduce executable (default=ON)
USE_SERIAL_GRIDDER    | Uses serial implementation of gridder (default=OFF)
USE_FFTSHIFT          | Explicitly performs FFT shifting of the image matrix (and beam if generated) after the FFT - results in slower imager (default=OFF)

When compiled with USE_FLOAT=ON, the large data arrays of real or complex numbers use the single-precision floating-point representation instead of the double-precision floating-point. 
In most systems the FLOAT type uses 4 bytes while DOUBLE uses 8 bytes. Thus, using FLOAT allows to reduce the memory usage and consequently the pipeline running time.
However, it reduces the algorithm's accuracy.

After building STP, the FFTW wisdom files shall be generated using the fftw-wisdom tool. By using these files, the FFT step executes much faster.
A CMake target is provided to generate the FFTW wisdom files. This target executes a script located in "project-root/scripts/fftw-wisdom" directory.
The location and filename of the generated FFTW wisdom files shall be provided in the input JSON configuration file.

By default, FFTW wisdom files are generated for the following matrix sizes: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536.
When executing the STP only the above image sizes can be used as input, since the FFTW wisdom files were generated only for these sizes. 
If different image sizes are required, the script for wisdom file generation shall be manually executed indicating the required image sizes.

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
 -n    | Do not generate fftw wisdom files

#### Manually
```sh
$ mkdir -p <path/to/build/directory>
$ cd <path/to/build/directory>
$ cmake -DCMAKE_BUILD_TYPE=Release -DUSE_FLOAT=OFF <path/to/project/src>
$ make all -j4
```
Then, generate the FFTW wisdom files using the available CMake target:
```sh
$ make fftwisdom
```
Alternatively, the generate_wisdom.sh script can be manually executed (see available options using --help):
```sh
$ cd <path/to/project>/scripts/fftw-wisdom
$ ./generate_wisdom.sh <OPTIONS>
```
For instance, when compiled in Release mode with -DUSE_FLOAT=ON run:
```sh
$ ./generate_wisdom.sh -r -f
```
When executed manually, the wisdom files are written into the wisdomfiles sub-directory created in the working directory.

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
Note that some benchmarks use the pre-generated FFTW wisdom files by fftw-wisdom tool. 
Please, be sure to run 'make fftwisdom' before the benchmarks.
If in-place fft configuration needs to benchmarked, additional wisdom files must be generated using 'make ifftwisdom'.

## Run STP Executables

STP provides three executables:
- reduce : Runs the entire pipeline (imager and source find stages);
- run_imagevis : Runs the imager stage (gridder + FFT + normalisation steps);
- run_sourcefind : Runs the source find stage (requires an input image).

These executables are located in the build-directory/reduce. 

### Reduce

The pipeline can be executed using:
```sh
./reduce [-d] [-l] <input-file-json> <input-file-npz> <output-file-json> <output-file-npz>
```

Argument | Usage  | Description
---------| -------| -------------
<input-file-json>  | required | Input JSON filename with configuration parameters (e.g. fastimg_oversampling_config.json).
<input-file-npz>   | required | Input NPZ filename with simulation data: uvw_lambda, vis, skymodel (e.g. simdata_nstep10.npz).
<output-file-json> | required | Output JSON filename for detected islands.
<output-file-npz>  | optional | Output NPZ filename for label map matrix (label_map).
-d, --diff | optional | Use residual visibilities - difference between 'input_vis' and 'model' visibilities.
-l, --log  | optional | Enable logger.

Example:
```sh
$ cd build-directory/reduce
$ ./reduce fastimg_oversampling_config.json simdata_nstep10.npz detected_islands.json -d -l
```
Note that the provided fastimg_oversampling_config.json file assumes that the pre-generated FFTW wisdom files are located in <build-directory>/wisdomfiles.
This is the default path of the FFTW wisdom files when generated by the 'make fftwisdom' command.
If a different directory was used, the wisdom file path in the JSON configuration file shall be properly setup.

### Run_imagevis

The imager can be executed using:

```sh
./run_imagevis [-d] [-l] <input-file-json> <input-file-npz> <output-file-npz>
```

Argument | Usage  | Description
---------| -------| -------------
<input-file-json>  | required | Input JSON filename with configuration parameters (e.g. fastimg_oversampling_config.json).
<input-file-npz>   | required | Input NPZ filename with simulation data: uvw_lambda, vis, skymodel (e.g. simdata_nstep10.npz).
<output-file-npz>  | optional | Output NPZ filename for image and beam matrices (image, beam).
-d, --diff | optional | Use residual visibilities - difference between 'input_vis' and 'model' visibilities.
-l, --log  | optional | Enable logger.

Example:
```sh
$ cd build-directory/reduce
$ ./run_imagevis fastimg_oversampling_config.json simdata_nstep10.npz dirty_image.npz -d -l
```

### Run_sourcefind

The source find procedure can be executed using:

```sh
./run_sourcefind  [-l] <input-file-json> <input-file-npz> <output-file-json> <output-file-npz>
```
                     
Argument | Usage  | Description
---------| -------| -------------
<input-file-json>  | required | Input JSON filename with configuration parameters (e.g. fastimg_oversampling_config.json).
<input-file-npz>   | required | Input NPZ filename with simulation data (image).
<output-file-json> | required | Output JSON filename for detected islands.
<output-file-npz>  | optional | Output NPZ filename for label map matrix (label_map).
-l, --log  | optional | Enable logger.

Example:
```sh
$ cd build-directory/reduce
$ ./run_sourcefind fastimg_oversampling_config.json dirty_image.npz detected_islands.json -l
```

## Run STP using python bindings

The C++ imager and source find functions of STP can be independently called from Python code, using the STP Python bindings module.
The following procedure shall be used:
- Import stp_python.so (located in the build directory) and other important modules, such as Numpy.
- Setup the variables and load input files required for the wrapper functions.
- Call the STP wrapper functions.

Example of STP Python bindings that runs the imager and source find functions:

```pyhton
import stp_python
import numpy as np

# Input simdata file must be located in the current directory
vis_filepath = 'simdata_nstep10.npz'

# This example is not computing residual visibilities. 'vis' component is directly used as input to the pipeline
with open(vis_filepath, 'rb') as f:
    npz_data_dict = np.load(f)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']
    vis_weights = npz_data_dict['snr_weights']

# Parameters of image_visibilities function
image_size = 8192
cell_size = 0.5
function = stp_python.KernelFunction.GaussianSinc
support = 3
trunc = support
kernel_exact = False
oversampling = 9
generate_beam = False
# Use stp_python.FFTRoutine.FFTW_ESTIMATE_FFT if wisdom files are not available
r_fft = stp_python.FFTRoutine.FFTW_WISDOM_FFT
# The FFTW wisdom files must be located in the current directory
fft_wisdom_filename = '../wisdomfiles/WisdomFile_rob8192x8192.fftw'

# Call image_visibilities
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(vis, 
        vis_weights, uvw_lambda, image_size, cell_size, function, 
        trunc, support, kernel_exact, oversampling, generate_beam, 
        r_fft, fft_wisdom_filename)

# Parameters of source_find function
detection_n_sigma = 50.0
analysis_n_sigma = 50.0
rms_est = 0.0
find_negative = True
sigma_clip_iters = 5
median_method = stp_python.MedianMethod.BINAPPROX  
# Other options: stp_python.MedianMethod.ZEROMEDIAN, stp_python.MedianMethod.BINMEDIAN, stp_python.MedianMethod.NTHELEMENT
gaussian_fitting = True
generate_labelmap = False
ceres_diffmethod = stp_python.CeresDiffMethod.AnalyticDiff_SingleResBlk 
# Other options: stp_python.CeresDiffMethod.AnalyticDiff, stp_python.CeresDiffMethod.AutoDiff_SingleResBlk, stp_python.CeresDiffMethod.AutoDiff
ceres_solvertype = stp_python.CeresSolverType.LinearSearch_LBFGS 
# Other options: stp_python.CeresSolverType.LinearSearch_BFGS, stp_python.CeresSolverType.TrustRegion_DenseQR

# Call source_find
islands = stp_python.source_find_wrapper(cpp_img, detection_n_sigma, 
            analysis_n_sigma, rms_est, find_negative, 
            sigma_clip_iters, median_method, gaussian_fitting, 
            generate_labelmap, ceres_diffmethod, ceres_solvertype)

# Print result
for i in islands:
   print(i)
   print()
```

## Run source finding module

The STP library can be used to run the source finding module independently using either the run_sourcefind executable or calling the stp_python.source_find_wrapper from the Python code, as previously described.
However, the input image must satisfy some requirements, in particular, it must be represented using the Double type and use the Fortran-style array order.
When these requirements are not meet, it performs an image conversion, which involves copying the entire image to a new array using double type and Fortran-style, degrading the performance of the algorithm.

Thus, for benchmarking purposes, the source finding executable and python bindings must be compiled using the double precision mode and the input image must be represented according to the referred requirements.
While using the double type for image representation should not be a problem, since numpy usually uses this type by default, setting the Fortran-style array order may require extra parameters when creating the array.
Another solution is to convert the previously created numpy array in C-style to Fortran-style using the np.asfortranarray() function from the Python code.


## Code profiling
 - Valgrind framework tools can be used to profile STP library: callgrind (function call history and instruction profiling), cachegrind (cache and branch prediction profiling) and massif (memory profiling).
 - For more accurate profiling STP and reduce shall be compiled in Release mode. However, if detailed profiling of source code lines is desired, debug information shall be added (-g option of gcc). This can be done using the "Release With Debug Information" mode (CMAKE_BUILD_TYPE=RelWithDebInfo), as it uses compiling optimizations while adding debug symbols.
 - When running valgrind with callgrind tool, add --separate-threads=yes in order to independently profile all threads.
 - Results of these tools are written out to a file (default name: \<tool\>.out.\<pid\>).
 - Callgrind and cachegrind output files can be analyzed using kcachegrind GUI application, while massif output file can be analyzed with massif-visualizer.
 - Callgrind usage example:
```sh
$ cd path/to/build/directory
$ cd reduce
$ valgrind --tool=callgrind --separate-threads=yes ./reduce fastimg_oversampling_config.json simdata_nstep10.npz detected_islands.json -d -l
$ kcachegrind callgrind.out.*
```
For memory checking purposes, a CMake target for valgrind that executes the test_pipeline_gaussiansinc test is provided:
```sh
$ cd path/to/build/directory
$ make valgrind
```
