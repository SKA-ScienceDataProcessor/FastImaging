#ifndef _GRIDDERPROTOTYPE_H
#define _GRIDDERPROTOTYPE_H

#include <armadillo>

using namespace std;
using namespace arma;

enum fftdirection { FORWARD, INVERSE };
extern int forward;
extern int inverse;

//
//	GLOBAL VARIABLES
//

// grid parameters.
extern double _cellSize;		// the angular size of each output pixel
extern double _uvCellSize;		// in units of lambda
extern int _uvPixels;
extern int _wPlanes;
extern double _fixedFrequency;
extern double _inputRA;
extern double _inputDEC;
extern double _outputRA;
extern double _outputDEC;

// samples.
extern int _numSamples;
extern mat _sample;

// channels.
extern int _numChannels;
extern vec _wavelength;
extern rowvec _frequency; //comment

// w-plane details.
extern vec _wPlaneMean;
extern vec _wPlaneMax;


// visibilities.
extern cx_rowvec _visibility;

// kernel parameters.
extern double _oversample;
extern int _support;
extern int _kernelSize;

// anti-aliasing kernel parameters.
extern int _aaSupport;
extern int _aaKernelSize;

// w-kernel parameters.
extern int _wSupport;
extern int _wKernelSize;

// kernel data.
extern cx_rowvec _kernel;

// store the anti-aliasing kernel (with no oversampling) because we will need to remove this from the image domain.
extern cx_rowvec _aaKernel;

// gridded data.
extern cx_rowvec _grid;

// deconvolution image.
extern cx_vec _deconvolutionImage;



//
//	CONSTANTS
//

// the input parameters from file gridder-params.
const char CELL_SIZE[] = "cell_size:";
const char PIXELS_UV[] = "pixels_uv:";
const char W_PLANES[] = "w_planes:";
const char OVERSAMPLE[] = "oversample:";
const char FREQUENCY[] = "frequency:";
const char INPUT_RA[] = "input-ra";
const char INPUT_DEC[] = "input-dec";
const char OUTPUT_RA[] = "output-ra";
const char OUTPUT_DEC[ ] = "output-dec";

// speed of light.
const long int CONST_C = 299792458;

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




//
//	GENERAL FUNCTIONS
//

int saveBitmap( char * pOutputFilename, cx_rowvec &grid );
void calculateKernelSize();
void doPhaseCorrection();
int generateKernel();
void getParameters();
int gridVisibilities( cx_rowvec &pGrid, cx_rowvec pVisibility, int pOversample, int pKernelSize, int pSupport,
                        cx_vec pKernel, int pWPlanes, mat pSample, int pNumSamples, int pNumChannels );
void performFFT( cx_rowvec &pGrid, int pSize, fftdirection pFFTDirection );
int loadData( char * pInputUVWFilename, char * pInputVisFilename, char * pInputChannelFilename);
int generateImageOfConvolutionFunction( char * pDeconvolutionFilename );
void generateAAKernel( cx_rowvec &pAAKernel, int pOversampledKernelSize, int pImageSize );
void updateKernel( cx_rowvec &pKernel, cx_rowvec &pImage, int pKernelSupport, int pImageSupport, int pOversampleI, int pOversampleJ, long startIndex);
void generateWKernel( cx_rowvec &pWKernel, int pImageSize, double pW, double pCellSizeRadians );

#endif // _GRIDDERPROTOTYPE_H
