// This file measures the performance of each function created on the cppGridderPrototype.cpp file

#include <benchmark/benchmark.h>
#include <cppGridderPrototype.h> // Original code had no header
#include <complex>
#include <memory>


/*! BM_performFFT_forward
 * \brief Benchmark performFFT function on forward computation.
 * \param[in] state benchmark state object.
 */
static void BM_performFFT_forward(benchmark::State& state)
{
    int image_area(state.range_x());
    std::unique_ptr< complex<double>[] > image_vector(new complex<double>[image_area]);

    // Fill in the image_vector array with known values
    for(int i = 0; i < image_area; ++i) {
        image_vector[i] = complex<double>(i, i);
    }

    int image_side(sqrt(image_area));
    while(state.KeepRunning() == true) {
        performFFT(image_vector.get(), image_side, FORWARD);
    }
    
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(image_area) * sizeof(complex<double>));
}

BENCHMARK(BM_performFFT_forward)->RangeMultiplier(2)->Range(8*8, 14188*14188); // approx. 3 GB worth of complex<double>


/*! BM_performFFT_inverse
 * \brief Benchmark performFFT function on inverse computation.
 * \param[in] state benchmark state object.
 */
static void BM_performFFT_inverse(benchmark::State& state)
{
    int image_area(state.range_x());
    std::unique_ptr< complex<double>[] > image_vector(new complex<double>[image_area]);

    // Fill in the array with known values
    for(int i = 0; i < image_area; ++i) {
        image_vector[i] = complex<double>(i, i);
    }

    int image_side(sqrt(image_area));
    while(state.KeepRunning() == true) {
        performFFT(image_vector.get(), image_side, INVERSE);
    }
    
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(image_area) * sizeof(complex<double>));
}

BENCHMARK(BM_performFFT_inverse)->RangeMultiplier(2)->Range(8*8, 14188*14188)->Complexity(); // approx. 3 GB worth of complex<double>


/*! BM_calculateKernelSize
 * \brief Benchmark calculateKernelSize function computation.
 * \param[in] state benchmark state object.
 */
static void BM_calculateKernelSize(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _wPlanes = 4;     
    _cellSize = 0.4;   
    _uvPixels = 2048; 
    
    while(state.KeepRunning() == true) {
        calculateKernelSize();
    }
}

BENCHMARK(BM_calculateKernelSize);


/*! BM_doPhaseCorrection
 * \brief Benchmark doPhaseCorrection function computation.
 * \param[in] state benchmark state object.
 */
static void BM_doPhaseCorrection(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _inputRA = 180;
    _inputDEC = 34;
    _outputRA = 180;
    _outputDEC = 34;
    _numSamples = 252720;
    _numChannels = 1;
    
    // Global values
    std::unique_ptr< VectorF[] > sample_vector(new VectorF[_numSamples]);
    _sample = sample_vector.get();
    std::unique_ptr< double[] > wavelength_vector(new double[_numChannels]);
    _wavelength = wavelength_vector.get();
    std::unique_ptr< complex<double>[] > visibility_vector(new complex<double>[_numSamples * _numChannels]);
    _visibility = visibility_vector.get();
    
    //Fill in the array _wavelength with known values
    for(int i(0); i < (_numChannels); i++) {
        _wavelength[i] = round(i);
    }
    
    //Fill in the VectorF _sample with known values
    for(int i(0); i < (_numSamples); i++) {
        _sample[i].u = round(i);
        _sample[i].v = round(i);
        _sample[i].w = round(i);
    }    
    
    //Fill in the _visibility vector with known values
    for(int i(0); i < (_numSamples*_numChannels); i++) {
        _visibility[i] = complex<double>(i, i);
    }
    
    while(state.KeepRunning() == true) {
        doPhaseCorrection();
    }
}

BENCHMARK(BM_doPhaseCorrection);


/*! BM_updateKernel
 * \brief Benchmark updateKernel function computation.
 * \param[in] state benchmark state object.
 */
static void BM_updateKernel(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _oversample = 1;
    _aaKernelSize = 9; 
    _aaSupport=3;
    
    int image_area(state.range_x());
    int image_side(sqrt(image_area));
    int image_support = (image_side - 1) / 2;
    
    std::unique_ptr< complex<double>[] > kernel_vector(new complex<double>[_aaKernelSize * _aaKernelSize]);
    _aaKernel = kernel_vector.get();
    std::unique_ptr< complex<double>[] > image_vector(new complex<double>[image_area]); 
    
    //Fill in the _aaKernel vector with known values
    for(int i(0); i < (_aaKernelSize * _aaKernelSize); i++) {
        _aaKernel[i] = complex<double>(i, i);
    }
    
    //Fill in the image_vector with known values
    for(int i(0); i < image_area; i++) {
        image_vector[i] = complex<double>(i, i);
    }
    
    while(state.KeepRunning() == true) {
        updateKernel( _aaKernel, image_vector.get(), _aaSupport, image_support, 0, 0 );
    }
}

BENCHMARK(BM_updateKernel)->RangeMultiplier(2)->Range(8*8, 14188*14188)->Complexity(); // approx. 3 GB worth of complex<double>


/*! BM_generateAAKernel
 * \brief Benchmark generateAAKernel function computation.
 * \param[in] state benchmark state object.
 */
static void BM_generateAAKernel(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _aaSupport = 3;
    _oversample = 1;
    
    int image_area(state.range_x());
    int image_side(sqrt(image_area));
    int oversampled_kernel_size = ((_aaSupport * 2) + 1) * _oversample;
    
    std::unique_ptr< complex<double>[] > kernel_vector(new complex<double>[image_area]);
    
    //Fill in the kernel_vector with known values
    for(int i(0); i < image_area; i++) {
        kernel_vector[i] = complex<double>(i, i);
    }
    
    while(state.KeepRunning() == true) {
        generateAAKernel( kernel_vector.get(), oversampled_kernel_size, image_side );
    }
}

BENCHMARK(BM_generateAAKernel)->RangeMultiplier(2)->Range(8*8, 14188*14188)->Complexity(); // approx. 3 GB worth of complex<double>


/*! BM_generateWKernel
 * \brief Benchmark generateWKernel function computation.
 * \param[in] state benchmark state object.
 */
static void BM_generateWKernel(benchmark::State& state)
{
    // The following value were extracted from the original test data
    _cellSize = 0.4;
    
    int image_area(state.range_x());
    int image_side(sqrt(image_area));
    double cell_size_radians((_cellSize / 3600.0) * (PI / 180.0));
    double var(-11092.7);

    std::unique_ptr< complex<double>[] > kernel_vector(new complex<double>[image_area]);
    
    //Fill in the kernel_vector with known values
    for(int i(0); i < image_area; i++) {
        kernel_vector[i] = complex<double>(i, i);
    }
    
    while(state.KeepRunning() == true) {
        generateWKernel( kernel_vector.get(), image_side, var, cell_size_radians );
    } 
}

BENCHMARK(BM_generateWKernel)->RangeMultiplier(2)->Range(8*8, 14188*14188)->Complexity(); // approx. 3 GB worth of complex<double>


/*! BM_generateKernel
 * \brief Calculate generateKernel() function performance
 * \param[in] state benchmark state object.
 */
static void BM_generateKernel(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _cellSize = 0.4;
    _oversample = 1;
    _support = 25;
    _aaSupport = 3;
    _wPlanes = 4;
    _kernelSize = 53;
    _aaKernelSize = 9;
    _uvPixels = 2048;
    
    std::unique_ptr< double[] > wplane_mean_vector(new double[_wPlanes]);
    _wPlaneMean = wplane_mean_vector.get();
    std::unique_ptr< double[] > wavelength_vector(new double[_numChannels]);
    _wavelength = wavelength_vector.get();    
    
    //Fill in the _wPlaneMean vector with known values
    for(int i(0); i < _wPlanes; i++) {
        _wPlaneMean[i] = round(i);
    }
    
    //Fill in the _wavelength vector  with known values
    for(int i(0); i < (_numChannels); i++) {
        _wavelength[i] = round(i);
    }
    
    while(state.KeepRunning() == true) {
        generateKernel();
    } 
}

BENCHMARK(BM_generateKernel);


/*! BM_gridVisibilities
 * \brief Calculate gridVisibilities() function performance
 * \param[in] state benchmark state object.
 */
static void BM_gridVisibilities(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _uvPixels = 2048;
    _kernelSize = (2 * _support) + 3;
    _numSamples = 252720;
    _numChannels = 1;
    _oversample = 1;
    _support = 25;
    _wPlanes = 4;
    _uvCellSize = 251.788094;
    
    int kernel_size(_kernelSize * _kernelSize * _oversample * _oversample * _wPlanes);
    
    std::unique_ptr< double[] > wplane_max_vector(new double[_wPlanes]);
    _wPlaneMax = wplane_max_vector.get();
    std::unique_ptr< VectorF[] > sample_vector(new VectorF[_numSamples]);
    _sample = sample_vector.get();
    std::unique_ptr< double[] > wavelength_vector(new double[_numChannels]);
    _wavelength = wavelength_vector.get();   
    
    std::unique_ptr< complex<double>[] > grid_kernel_vector(new complex<double>[_uvPixels*_uvPixels]);
    std::unique_ptr< complex<double>[] > visibility_kernel_vector(new complex<double>[_numSamples * _numChannels]);
    std::unique_ptr< complex<double>[] > kernel_vector(new complex<double>[kernel_size]);
    
    //Fill in the _wPlaneMax vector  with known values
    for(int i(0); i < _wPlanes; i++) {
        _wPlaneMax[i] = round(i);
    }   
    
    //Fill in the _sample vector with known values
    for(int i=0; i<_numSamples; i++) {
        _sample[i].u = i;
        _sample[i].v = i;
        _sample[i].w = i;
    }     
    
    //Fill in the _sample vector with known values
    for(int i(0); i < (_numChannels); i++) {
        _wavelength[i] = round(i);
    } 
    
    //Fill in the grid vector with known values
    for(int i=0; i < (_uvPixels * _uvPixels); i++) {
        grid_kernel_vector[i] = complex<double>(i,i);
    }

    //Fill in the visibility_kernel_vector vector with known values
    for(int i=0; i< (_numSamples*_numChannels); i++) {
        visibility_kernel_vector[i] = complex<double>(i,i);
    }    
    
    //Fill in the kernel_vector vector with known values
    for(int i=0; i< kernel_size ; i++) {
        kernel_vector[i] = complex<double>(i,i);
    }

    while(state.KeepRunning() == true) {
        gridVisibilities( grid_kernel_vector.get(), visibility_kernel_vector.get(), _oversample, _kernelSize, _support, kernel_vector.get(), _wPlanes, _sample, _numSamples, _numChannels );
    }
}

BENCHMARK(BM_gridVisibilities);


/*! BM_generateImageOfConvolutionFunction
 * \brief Calculate generateImageOfConvolutionFunction() function performance
 * \param[in] state benchmark state object.
 */
static void BM_generateImageOfConvolutionFunction(benchmark::State& state)
{
    // The following values were extracted from the original test data
    _uvPixels = 2048;
    _aaKernelSize = 9;
    _aaSupport= 3;
    _numChannels = 1;
    _uvCellSize = 251.788094;
    
    char pDeconvolutionFilename[20] = "Deconvolution";
    
    std::unique_ptr< complex<double>[] > kernel_vector(new complex<double>[_aaKernelSize * _aaKernelSize]);
    _aaKernel = kernel_vector.get();
    std::unique_ptr< double[] > wavelength_vector(new double[_numChannels]);
    _wavelength = wavelength_vector.get(); 
    std::unique_ptr< double[] > wplane_max_vector(new double[_wPlanes]);
    _wPlaneMax = wplane_max_vector.get();
    
    //Fill in the _aaKernel vector with known values
    for(int i(0); i < (_aaKernelSize * _aaKernelSize); i++) {
        _aaKernel[i] = complex<double>(i, i);
    }    
    
    //Fill in the _wavelength vector with known values
    for(int i(0); i < (_numChannels); i++) {
        _wavelength[i] = round(i);
    } 
    
    //Fill in the _wPlaneMax vector with known values
    for(int i(0); i < _wPlanes; i++) {
        _wPlaneMax[i] = round(i);
    }
 
    while(state.KeepRunning() == true) {
        generateImageOfConvolutionFunction( pDeconvolutionFilename );
    }
}    

BENCHMARK(BM_generateImageOfConvolutionFunction);

// Run the tests
BENCHMARK_MAIN();
