#ifndef CPPGRIDDERPROTOTYPE_H
#define CPPGRIDDERPROTOTYPE_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include <math.h>
#include <string.h>
#include <fftw3.h>
#include <time.h>

using namespace std;

//
//	FORWARDS DECLARATIONS
//

bool saveBitmap( char * pOutputFilename );

//
//	CONSTANTS
//

// the input parameters from file gridder-params.
const char CELL_SIZE[] = "cell_size:";
const char PIXELS_UV[] = "pixels_uv:";
const char W_PLANES[] = "w_planes:";
const char OVERSAMPLE[] = "oversample:";
const char INPUT_RA[] = "input-ra:";
const char INPUT_DEC[] = "input-dec:";
const char OUTPUT_RA[] = "output-ra:";
const char OUTPUT_DEC[] = "output-dec:";

// speed of light.
const long int CONST_C = 299792458;
const double PI = 3.141592654;

// bitmap file header positions.
const int BIT_CONST = 0x00;
const int MAP_CONST = 0x01;
const int IMAGE_SIZE = 0x02;
const int RESERVED = 0x06;
const int FILE_HEADER_SIZE = 0x0A;
const int BITMAP_INFO_HEADER = 0x0E;
const int IMAGE_WIDTH = 0x12;
const int IMAGE_HEIGHT = 0x16;
const int COLOUR_PLANES = 0x1A;
const int BIT_COUNT = 0x1C;
const int COMPRESSION_TYPE = 0x1E;
const int COLOURS_USED = 0x2E;
const int SIGNIFICANT_COLOURS = 0x32;

// enumerated types.
enum fftdirection { FORWARD, INVERSE };

//
//	STRUCTURES
//

// vector with floats. Can be used either as a 2 or 3 element vector.
struct VectorF
{
	double u;
	double v;
	double w;
};
typedef struct VectorF VectorF;

//
//	GLOBAL VARIABLES
//

// grid parameters.
double _cellSize = 0;		// the angular size of each output pixel
double _uvCellSize = 0;		// in units of lambda
int _uvPixels = 0;
int _wPlanes = 0;
double _inputRA = 0;
double _inputDEC = 0;
double _outputRA = 0;
double _outputDEC = 0;
	
// samples.
int _numSamples = 0;
VectorF * _sample = NULL;

// channels.
int _numChannels = 0;
double * _wavelength = NULL;
	
// w-plane details.
double * _wPlaneMean = NULL;
double * _wPlaneMax = NULL;
	
// visibilities.
complex<double> * _visibility = NULL;
	
// kernel parameters.
double _oversample = 0;
int _support = 0;
int _kernelSize = 0;

// anti-aliasing kernel parameters.
int _aaSupport = 0;
int _aaKernelSize = 0;

// w-kernel parameters.
int _wSupport = 0;
int _wKernelSize = 0;

// kernel data.
complex<double> * _kernel = NULL;
	
// store the anti-aliasing kernel (with no oversampling) because we will need to remove this from the image domain.
complex<double> * _aaKernel = NULL;
	
// gridded data.
complex<double> * _grid = NULL;
	
// deconvolution image.
complex<double> * _deconvolutionImage = NULL;

//
//	GENERAL FUNCTIONS
//

//
//	performFFT()
//
//	CJS: 11/08/2015
//
//	Make a dirty image by inverse FFTing the gridded visibilites.
//

void performFFT( complex<double> * pGrid, int pSize, fftdirection pFFTDirection );

//
//	quickSort()
//
//	CJS: 14/01/2016
//
//	Sort a list of double values.
//

void quickSort( double * pValues, long int pLeft, long int pRight );

//
//	calculateKernelSize()
//
//	CJS: 07/08/2015
//
//	Calculate some values such as support, kernel size and w-cell size. We need to work out the maximum baseline length first. The support is given
//	by the square root of the number of uv cells that fit into the maximum baseline, multiplied by 1.5. The w-cell size is given by the maximum
//	baseline length divided by the number of w-planes, multiplied by two.
//

void calculateKernelSize();

//
//	doPhaseCorrection()
//
//	CJS: 12/08/2015
//
//	Phase correct all the visibilities using the PhaseCorrection class.
//

void doPhaseCorrection();

//
//	getTime()
//
//	CJS: 16/11/2015
//
//	Get the elapsed time.
//

float getTime( struct timespec start, struct timespec end );

//
//	updateKernel()
//
//	CJS: 29/04/2015
//
//	Updates the kernel function by extracting the centre part of an image, taking into account the oversample parameter.
//
//	A-----B-----C-----D-----E-----F
//	|     |     |     |     |     |
//	|     |   c-c-c-d-d-d-e |     |
//	|     |   | | | | | | | |     |
//	|     |   i-i-i-j-j-j-k |     |
//	|     |   | | |\|/| | | |     |
//	G-----H---i-I-i-j-J-j-k-K-----L
//	|     |   | | |/|\| | | |     |
//	|     |   i-i-i-j-j-j-k |     |
//	|     |   | | | | | | | |     |
//	|     |   o-o-o-p-p-p-q |     |
//      |     |     |     |     |     |
//	M-----N-----O-----P-----Q-----R
//
//	The diagram show the UV grid (large squares), and x3 oversampled kernel (small squares). The position of the visibility
//	is indicated by the diagonal cross-hairs, and the oversampled kernel is centred at this point. The letters A-R identify
//	the UV grid point, and for each grid point we average all the cells of the oversampled kernel which are indicated by the
//	same letter (in lower case). i.e. for UV grid cell C there are 3 (out of a possible 9) kernel cells which are summed and
//	then divided by 9 to get the kernel value at this UV cell point.
//
//	Before we start gridding we take our oversampled kernel and extract from it NxN (where N is the oversample
//	parameter) low-resolution kernels, which are stored in memory ready to be used during gridding.
//

void updateKernel( complex<double> * pKernel, complex<double> * pImage, int pKernelSupport, int pImageSupport, int pOversampleI, int pOversampleJ );

//
//	generateAAKernel()
//
//	CJS: 29/04/2015
//
//	Generate the anti-aliasing kernel.
//

void generateAAKernel( complex<double> * pAAKernel, int pOversampledKernelSize, int pImageSize );

//
//	generateWKernel()
//
//	CJS: 29/04/2016
//
//	Generate the W-kernel.
//

void generateWKernel( complex<double> * pWKernel, int pImageSize, double pW, double pCellSizeRadians );

//
//	generateKernel()
//
//	CJS: 16/12/2015
//
//	Generates the convolution function. A separate kernel is generated for each w-plane, and for each oversampled intermediate
//	grid position.
//

bool generateKernel();

//
//	getParameters()
//
//	CJS: 07/08/2015
//
//	Load the following parameters from the parameter file gridder-params: uv cell size, uv grid size, # w-planes, oversample.
//

void getParameters();

//
//	gridVisibilities()
//
//	CJS: 10/08/2015
//
//	Produce an image of gridding visibilities by convolving the complex visibilities with the kernel function.
//

bool gridVisibilities( complex<double> * pGrid, complex<double> * pVisibility, int pOversample, int pKernelSize, int pSupport,
			complex<double> * pKernel, int pWPlanes, VectorF * pSample, int pNumSamples, int pNumChannels );

//
//	loadData()
//
//	CJS: 07/08/2015
//
//	Load the data from the measurement set. We need to load a list of sample (the uvw coordinates), a list of channels (we need the frequency of each channel),
//	and a list of visibility (we should have one visibility for each sample/channel combination).
//

bool loadData( char * pInputUVWFilename, char * pInputVisFilename, char * pInputChannelFilename );

//
//	saveBitmap()
//
//	CJS: 10/08/2015
//
//	Save the current grid as a bitmap file.
//

bool saveBitmap( char * pOutputFilename, complex<double> * pGrid );

//
//	generateImageOfConvolutionFunction()
//
//	CJS: 05/01/2016
//
//	Generate the deconvolution function by gridding a single source at u = 0, v = 0, and FFT'ing.
//

bool generateImageOfConvolutionFunction( char * pDeconvolutionFilename );

#endif // CPPGRIDDERPROTOTYPE_H
