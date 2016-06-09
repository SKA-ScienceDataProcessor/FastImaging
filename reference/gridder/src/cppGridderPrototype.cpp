#include "cppGridderPrototype.h"
#include <fftw3.h>
// my phase correction code.
#include "phasecorrection.h"

double _cellSize = 0;
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


using namespace std;

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

void performFFT( complex<double> * pGrid, int pSize, fftdirection pFFTDirection )
{
#ifndef TESTING
	printf( "performing fft.....\n" );
#endif
           
           
	// reserve some FFTW memory for the image, and generate a plan.
	fftw_complex * gridFFT = (fftw_complex *) fftw_malloc( pSize * pSize * sizeof( fftw_complex ) );
	fftw_plan fftPlan = fftw_plan_dft_2d( pSize, pSize, gridFFT, gridFFT, (pFFTDirection == INVERSE ? FFTW_BACKWARD : FFTW_FORWARD), FFTW_MEASURE );
	// the floor and ceiling pivots will be the same if we have an even image size, and forward and inverse FFT shifts will
	// be identical. but for uneven image sizes we must be more careful.
	int floorPivot = (int) floor( (double)pSize / 2.0 );
	int ceilPivot = (int) ceil( (double)pSize / 2.0 );
	int iFrom, jFrom;
	
	// do inverse FFT shift before FFTing.
	for ( int i = 0; i < pSize; i++ )
		for ( int j = 0; j < pSize; j++ )
		{
			if (i < floorPivot)
				iFrom = i + ceilPivot;
			else
				iFrom = i - floorPivot;
			if (j < floorPivot)
				jFrom = j + ceilPivot;
			else
				jFrom = j - floorPivot;
			memcpy( &gridFFT[ (j * pSize) + i ], &pGrid[ (jFrom * pSize) + iFrom ], sizeof( complex<double> ) );
		}
		
	
	// execute the fft.
	fftw_execute( fftPlan );
        
     
        
	// do forward FFT shift before FFTing.
	for ( int i = 0; i < pSize; i++ )
		for ( int j = 0; j < pSize; j++ )
		{
			if (i < ceilPivot)
				iFrom = i + floorPivot;
			else
				iFrom = i - ceilPivot;
			if (j < ceilPivot)
				jFrom = j + floorPivot;
			else
				jFrom = j - ceilPivot;
			memcpy( &pGrid[ (j * pSize) + i ], &gridFFT[ (jFrom * pSize) + iFrom ], sizeof( complex<double> ) );
		}
		

		
	// destroy the FFT plan.
	fftw_destroy_plan( fftPlan );
	fftw_cleanup();
	fftw_free( gridFFT );
	
} // performFFT

//
//	quickSort()
//
//	CJS: 14/01/2016
//
//	Sort a list of double values.
//

void quickSort( double * pValues, long int pLeft, long int pRight )
{
	
	long int i = pLeft, j = pRight;
	double tmp;
	double pivot = pValues[ (pLeft + pRight) / 2 ];
	
	// partition.
	while (i <= j)
	{
		while (pValues[ i ] < pivot)
			i = i + 1;
		while (pValues[ j ] > pivot)
			j = j - 1;
		if (i <= j)
		{
			tmp = pValues[ i ];
			pValues[ i ] = pValues[ j ];
			pValues[ j ] = tmp;
			i = i + 1;
			j = j - 1;
		}
	}
	
	// recursion.
	if (pLeft < j)
		quickSort( pValues, pLeft, j );
	if (i < pRight)
		quickSort( pValues, i, pRight );
	
} // quickSort

//
//	calculateKernelSize()
//
//	CJS: 07/08/2015
//
//	Calculate some values such as support, kernel size and w-cell size. We need to work out the maximum baseline length first. The support is given
//	by the square root of the number of uv cells that fit into the maximum baseline, multiplied by 1.5. The w-cell size is given by the maximum
//	baseline length divided by the number of w-planes, multiplied by two.
//

void calculateKernelSize()
{
	
   //     cout << "_uvPixels = " << _uvPixels << endl;
   //     cout << "_cellSize = " << _cellSize << endl;
    
	// calculate the size of each uv pixel.
	_uvCellSize = (1 / (_uvPixels * (_cellSize / 3600) * (PI / 180)));
	
	// get the maximum baseline length in metres, and the lowest frequency (index 0 of the array).
	//double maxBaseline = 1;
	//double wavelength = 0;
	//if (_numChannels > 0)
	//	wavelength = _wavelength[ _numChannels - 1 ];
	
	// loop through the samples to find the maximum baseline length.
	//for ( long int i = 0; i < _numSamples; i++ )
	//	if (sqrt( pow( _sample[i].u, 2 ) + pow( _sample[i].v, 2 ) + pow( _sample[i].w, 2 ) ) > maxBaseline)
	//		maxBaseline = sqrt( pow( _sample[i].u, 2 ) + pow( _sample[i].v, 2 ) + pow( _sample[i].w, 2 ) );
	
	// convert the maximum baseline length into units of lambda.
	//double maxBaselineLambda = maxBaseline / wavelength;
        
       
		
	// if we are using w-projection, we need to determine the distribution of w-planes.
	if (_wPlanes > 1)
	{
	
		// build a list of w-values in order to determine the distribution of w-planes.
		double * wValue = (double *) malloc( _numSamples * _numChannels * sizeof( double ) );
		for ( long int i = 0; i < _numSamples; i++ )
		
			// calculate the w-values for each channel.
			for ( int j = 0; j < _numChannels; j++ )
				wValue[ (i * _numChannels) + j ] = _sample[i].w / _wavelength[ j ];
	
		// sort these w-values.
		quickSort( wValue, 0, (_numSamples * _numChannels) - 1 );
				
		// divide the w-values up into the required number of planes. the planes are therefore not
		// evenly spaced, but spaced to match the density distribution of w-values.
		int visibilitiesPerWPlane = _numSamples * _numChannels / _wPlanes;
				
		// create some memory for the mean and maximum W.
		_wPlaneMean = (double *) malloc( _wPlanes * sizeof( double ) );
		_wPlaneMax = (double *) malloc( _wPlanes * sizeof( double ) );
				
		// we have N visibilities per plane. for each plane find the average W, and the maximum W.
		for ( int i = 0; i < _wPlanes; i++ )
		{
			
			double totalW = 0;
			
			// determine the range of visibilities that fall into this w-plane.
			long int minVisibility = i * visibilitiesPerWPlane;
			long int maxVisibility = minVisibility + visibilitiesPerWPlane;
			if (i == _wPlanes - 1)
				maxVisibility = _numSamples * _numChannels;
			_wPlaneMax[ i ] = wValue[ maxVisibility - 1 ];
			
			// add up the w-values for these visibilities.
			for ( int j = minVisibility; j < maxVisibility; j++ )
				totalW = totalW + wValue[ j ];
			
			// calculate the mean w-value.
			_wPlaneMean[ i ] = totalW / (double)(maxVisibility - minVisibility);
			
		}
#ifndef TESTING
		printf( "Maximum w-value: %f lambda\n", _wPlaneMax[ _wPlanes - 1 ] );
#endif
		// free w-value memory.
		free( wValue );
			
	}
		
	// set the properties of the anti-aliasing kernel.
	_aaSupport = 3;
	_aaKernelSize = (2 * _aaSupport) + 3;
	
	// set the properties of the w-kernel.
	_wSupport = 0;
	if (_wPlanes > 1)
		_wSupport = 25;
	_wKernelSize = (2 * _wSupport) + 1;
	
	// calculate the support, which should be either the w-support or the aa-support, whichever is larger.
	if (_aaSupport > _wSupport)
		_support = _aaSupport;
	else
		_support = _wSupport;
	
	// calculate the kernel size. thix would normally be 2.S + 1, but I add an extra couple of cells in order that
	// my implementation of oversampling works correctly.
	_kernelSize = (2 * _support) + 3;
	
#ifndef TESTING
        
	printf( "_support = %i\n", _support );
	printf( "_aaSupport = %i\n", _aaSupport );
	printf( "_wSupport = %i\n", _wSupport );
	//printf( "maxBaseline = %f m\n", maxBaseline );
	//printf( "maxBaselineLambda = %f\n", maxBaselineLambda );
	printf( "_cellSize = %f arcsec, %1.12f rad\n", _cellSize, (_cellSize / 3600) * (PI / 180) );
	printf( "_uvCellSize = %f\n", _uvCellSize );
	//printf( "frequency = %f Hz\n", frequency ); 
#endif
        
	
} // calculateKernelSize

//
//	doPhaseCorrection()
//
//	CJS: 12/08/2015
//
//	Phase correct all the visibilities using the PhaseCorrection class.
//

void doPhaseCorrection()
{
	
    
	const char J2000[] = "J2000";
	const double INTERVAL = 0.05; // progress is updated at these % intervals.
	
	PhaseCorrection phaseCorrection;
	
	// set up the coordinate system.
	phaseCorrection.inCoords.longitude = _inputRA;
	phaseCorrection.inCoords.latitude = _inputDEC;
	strcpy( phaseCorrection.inCoords.epoch, J2000 );
	
	phaseCorrection.outCoords.longitude = _outputRA;
	phaseCorrection.outCoords.latitude = _outputDEC;
	strcpy( phaseCorrection.outCoords.epoch, J2000 );
	
	phaseCorrection.uvProjection = false;
	
	// initialise phase rotation matrices.
	phaseCorrection.init();
	
	int fraction = -1;
#ifndef TESTING      
	printf( "\nphase rotating visibilities.....\n" );
#endif
	// loop through all the visibilities.
	for ( int i = 0; i < _numSamples; i++ )
	{
		
		// get the rotated uvw coordinate.
		Vector uvwIn;
		uvwIn.x = _sample[i].u;
		uvwIn.y = _sample[i].v;
		uvwIn.z = _sample[i].w;
		phaseCorrection.uvwIn = uvwIn;
		phaseCorrection.rotate();
		double phase = phaseCorrection.phase;
		VectorF uvwOut;
		uvwOut.u = phaseCorrection.uvwOut.x;
		uvwOut.v = phaseCorrection.uvwOut.y;
		uvwOut.w = phaseCorrection.uvwOut.z;
		_sample[i] = uvwOut;
		
		// loop through all the channels.
		for ( int j = 0; j < _numChannels; j++ )
		{
			
			double wavelength = _wavelength[ j ];
		
			// calculate the phasor.
			complex<double> phasor = complex<double>( cos( 2 * PI * phase / wavelength ),
									sin( 2 * PI * phase / wavelength ) );
			
			// multiply phasor by visibility.
			_visibility[ (i * _numChannels) + j ] = _visibility[ (i * _numChannels) + j ] * phasor;
			
		}
		
		// display progress.
		if ((double)i / (double)_numSamples >= ((double)(fraction + 1) * INTERVAL))
		{
			fraction = fraction + 1;
#ifndef TESTING
        		printf( "%i%%.", (int)(fraction * INTERVAL * 100) );
                        fflush( stdout );
#endif
		}
		
	}
#ifndef TESTING
	printf( "100%%\n" );
#endif
	
} // doPhaseCorrection

//
//	getTime()
//
//	CJS: 16/11/2015
//
//	Get the elapsed time.
//

float getTime( struct timespec start, struct timespec end )
{

	return ((float)(end.tv_sec - start.tv_sec) * 1000.0) + ((float)(end.tv_nsec - start.tv_nsec) / 1000000.0);

} // getTime

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

void updateKernel( complex<double> * pKernel, complex<double> * pImage, int pKernelSupport, int pImageSupport, int pOversampleI, int pOversampleJ )
{
	
	// get the size of the kernel, the image, and the oversampled kernel. the kernel size would normally be related
	// to the support, S, by 2.S + 1, but we add a couple of extra pixels to make sure oversampling works properly.
	int kernelSize = (pKernelSupport * 2) + 3;
	int imageSize = (pImageSupport * 2) + 1;
	int oversampledKernelSize = ((pKernelSupport * 2) + 1) * _oversample;
	
	// calculate the support of the oversampled kernel.
	int oversampledKernelSupport = (int)floor( (double)(oversampledKernelSize - 1) / 2.0 );
	
	// for oversampling factor N, we build our kernel by looping over the oversampled kernel with indexes i and j that are
	// incremented by N at each iteration. each iteration involves averaging the pixels in the region defined by
	// <i - N/2, j - N/2> and <i + N/2, j + N/2>. our starting position is therefore N/2 pixels relative to <i,j>.
	//int start = -(int)floor( (double)_oversample / 2.0 );

	for ( int i = 0; i < kernelSize; i++ )
		for ( int j = 0; j < kernelSize; j++ )
		{
			
			complex<double> total = complex<double>( 0, 0 );
			for ( int averageI = 0; averageI < _oversample; averageI++ )
				for ( int averageJ = 0; averageJ < _oversample; averageJ++ )
				{
					
					// get the coordinates within the oversampled kernel.
					//int kernelI = start + (i * _oversample) - pOversampleI + averageI;
					//int kernelJ = start + (j * _oversample) - pOversampleJ + averageJ;
					int kernelI = (i * _oversample) - pOversampleI + averageI;
					int kernelJ = (j * _oversample) - pOversampleJ + averageJ;
					
					// get the coordinates within the whole image.
					int imageI = kernelI + pImageSupport - oversampledKernelSupport;
					int imageJ = kernelJ + pImageSupport - oversampledKernelSupport;

					// check that we haven't gone outside the image dimensions or the oversampled kernel dimensions.
					if (imageI >= 0 && imageI < imageSize && imageJ >= 0 && imageJ < imageSize &&
						kernelI >= 0 && kernelI < oversampledKernelSize && kernelJ >= 0 && kernelJ < oversampledKernelSize)
						
						// add this pixel to the total.
						total = total + pImage[ (imageJ * imageSize) + imageI ];
					
				}
				
			// divide the total by the number of oversampled pixels, and update the output kernel.
			pKernel[ (j * kernelSize) + i ] = total / pow( (double)_oversample, 2 );
			
		}
	
} // updateKernel

//
//	generateAAKernel()
//
//	CJS: 29/04/2015
//
//	Generate the anti-aliasing kernel.
//

void generateAAKernel( complex<double> * pAAKernel, int pOversampledKernelSize, int pImageSize )
{
	
	// constants.
	const int NP = 4;
	const int NQ = 2;
	
	// data for spheroidal gridding function.
	double dataP[2][5] = {	{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
				{4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}	};
	double dataQ[2][3] = {	{1.0, 8.212018e-1, 2.078043e-1},
				{1.0, 9.599102e-1, 2.918724e-1} };
	
	// calculate the support of the image and the oversampled kernel. we use an integer for the image support because
	// we know the image size is odd, which implies an integer support size.
	double oversampledSupport = ((double)pOversampledKernelSize - 1.0) / 2.0;
	int imageSupport = ((double)pImageSize - 1.0) / 2.0;
				
	// now loop through the i and j (u and v) coordinates of the kernel.
	for ( int i = 0; i < pOversampledKernelSize; i++ )
	{
					
		// get the i coordinates for the whole image.
		int imageI = i + imageSupport - (int)ceil( oversampledSupport );
						
		// calculate x-offset from centre of kernel.
		double x = (double)i - oversampledSupport;
					
		// calculate the component of r^2 in the u direction.
		double rSquaredI = pow( x, 2 );
					
		for ( int j = 0; j < pOversampledKernelSize; j++ )
		{
						
			// get the j coordinates for the whole image.
			int imageJ = j + imageSupport - (int)ceil( oversampledSupport );
						
			// calculate y-offset from centre of kernel/image.
			double y = (double)j - oversampledSupport;
						
			// get kernel pointer.
			complex<double> * aaKernelPtr = &pAAKernel[ (imageJ * pImageSize) + imageI ];
						
			// calculate r, which is the offset from the centre of the kernel in dimensional grid units.
			double r = sqrt( rSquaredI + pow( y, 2 ) );
			
			// get the distance (0 to 1) from the centre of the kernel to the first minimum of the prolate spheroidal.
			double spheroidRadius = r / (double)oversampledSupport;
					
			// now, calculate the anti-aliasing kernel.
			double val = 0;
						
			// only calculate a kernel value if this pixel is within the required radius from the centre.
			if (spheroidRadius <= 1)
			{
						
				int part = 1;
				double radiusEnd = 1.0;
				if (spheroidRadius < 0.75)
				{
					part = 0;
					radiusEnd = 0.75;
				}
							
				double delRadiusSq = (spheroidRadius * spheroidRadius) - (radiusEnd * radiusEnd);
							
				double top = 0;
				for (int k = 0; k < NP; k++)
					top = top + (dataP[ part ][ k ] * pow( delRadiusSq, k ));
							
				double bottom = 0;
				for (int k = 0; k < NQ; k++)
					bottom = bottom + (dataQ[ part ][ k ] * pow( delRadiusSq, k ));
					
				if (bottom != 0)
					val = top / bottom;
							
				// the gridding function is (1 - spheroidRadius^2) x gridsf
				val = val * (1 - (spheroidRadius * spheroidRadius));
							
			}
				
			// update the appropriate pixel in the anti-aliasing kernel.
			*aaKernelPtr = complex<double>( val, 0 );
					
		}
					
	}
	
} // generateAAKernel

//
//	generateWKernel()
//
//	CJS: 29/04/2016
//
//	Generate the W-kernel.
//

void generateWKernel( complex<double> * pWKernel, int pImageSize, double pW, double pCellSizeRadians )
{
	
	// calculate the image support. this will be an integer since the image size is odd.
	int imageSupport = (pImageSize - 1) / 2;
	
	// calculate the cell size squared.
	double cellSizeRadiansSquared = pow( pCellSizeRadians, 2 );
	
	for ( int i = 0; i < pImageSize; i++ )
	{
				
		// calculate the x offset from the centre of the image.
		double x = pow( (double)(i - imageSupport), 2 );
		
		for ( int j = 0; j < pImageSize; j++ )
		{
	
			// get kernel pointer.
			complex<double> * wKernelPtr = &pWKernel[ (j * pImageSize) + i ];
				
			// calculate the y offset from the centre of the image.
			double y = pow( (double)(j - imageSupport), 2 );
				
			// calculate r^2 (dist from centre of image squared).
			double rSquared = x + y;
				
			// convert r^2 from pixels to radians.
			double rSquaredRadians = rSquared * cellSizeRadiansSquared;
				
			// scale r^2 by the oversampling parameter O, we want to shrink our function by a factor of O so that
			// the kernel in the uv domains appears O times larger.
			rSquaredRadians *= (double)(_oversample * _oversample);
				
			// calculate kernel values,
			*wKernelPtr = complex<double>(	cos( 2 * PI * pW * (sqrt( 1.0 - rSquaredRadians ) - 1.0) ) / sqrt( 1.0 - rSquaredRadians ),
							sin( 2 * PI * pW * (sqrt( 1.0 - rSquaredRadians ) - 1.0) ) / sqrt( 1.0 - rSquaredRadians ) );
			
		}
		
	}
	
} // generateWKernel

//
//	generateKernel()
//
//	CJS: 16/12/2015
//
//	Generates the convolution function. A separate kernel is generated for each w-plane, and for each oversampled intermediate
//	grid position.
//

bool generateKernel()
{
	bool ok = true;
	double cellSizeRadians = (_cellSize / 3600.0) * (PI / 180.0);
	
	// calculate the size of the images we want to use to construct our kernels. we choose to make this number odd, so
	// we will use the global grid size if it is odd, otherwise we use grid size - 1 pixel.
	int imageSize = _uvPixels;
	if ((double)imageSize / 2.0 == floor( (double)imageSize / 2.0 ))
		imageSize = imageSize - 1;
				
	// image support. this is the centre of the image, and since image size is odd then the support will be an integer.
	int imageSupport = (imageSize - 1) / 2;
	
	// calculate our oversampled kernel size. this is equal to the required final kernel size multiplied by the oversampling
	// parameter.
	int oversampledKernelSize = ((_support * 2) + 1) * _oversample;
	
	// reserve some memory for the kernels.
	_kernel = (complex<double> *) malloc( _kernelSize * _kernelSize * _oversample * _oversample * _wPlanes * sizeof( complex<double> ) );
		
	// reserve some memory for the global anti-aliasing kernel. this is used to remove the kernel convolution.
	_aaKernel = (complex<double> *) malloc( _aaKernelSize * _aaKernelSize * sizeof( complex<double> ) );
        
	// reserve some memory for a w-kernel.
	complex<double> * wKernel = (complex<double> *) malloc( imageSize * imageSize * sizeof( complex<double> ) );
	
	// reserve some memory for a aa-kernel.
	complex<double> * aaKernel = (complex<double> *) malloc( imageSize * imageSize * sizeof( complex<double> ) );
	
	// calculate the oversampled AA kernel size. if this figure is even, subtract one. we MUST have an odd kernel size
	// in order that our kernel function, which is positioned on the centre of a pixel rather than between pixels, is centred
	// within the kernel.
	int oversampledAAKernelSize = ((_aaSupport * 2) + 1) * _oversample;
	if ((float)oversampledAAKernelSize / 2.0 == floor( (float)oversampledAAKernelSize / 2.0 ))
		oversampledAAKernelSize = oversampledAAKernelSize - 1;
		
	// generate the anti-alliasing kernel in image [aaKernel] of size [imageSize].
	// the kernel will be prolate spheroidal function of width [oversampledKernelSize]
	generateAAKernel( aaKernel, oversampledAAKernelSize, imageSize );
	
	// copy the anti-aliasing kernel from its temporary home into the global anti-aliasing kernel. it will be
// 	// used later to deconvolve the dirty image.
	updateKernel( _aaKernel, aaKernel, _aaSupport, imageSupport, 0, 0 );
        
	// if we are using w-projection then we will need to combine the AA-kernel with the W-kernel, and, for this, we will
	// need the AA-kernel to be in the image domain. FFT the AA-kernel.
	if (_wPlanes > 1)
		performFFT( aaKernel, imageSize, INVERSE );
	// calculate separate kernels for each w-value.
	for ( int w_plane = 0; w_plane < _wPlanes; w_plane++ )
	{
			
		// calculate kernel offset.
		long int kernelOffsetW = w_plane * _kernelSize * _kernelSize * _oversample * _oversample;
		
		if (_wPlanes > 1)
		{
			
			// generate a w-kernel with size [imageSize] in [wKernel].
                    
                       
			generateWKernel( wKernel, imageSize, _wPlaneMean[ w_plane ], cellSizeRadians );
			
			// now we need to convolve the w-kernel and the AA-kernel. we don't do this is are not
			// using w-projection. we multiply the kernels in the image domain.
			for ( int i = 0; i < imageSize; i++ )
				for ( int j = 0; j < imageSize; j++ )
					wKernel[ (j * imageSize) + i ] *= aaKernel[ (j * imageSize) + i ];
				
                          cout << "perform wKernel fft *********"<<wKernel[0] << endl;      
                                
			// perform a 2D fft to move the kernel into the uv domain.
			performFFT( wKernel, imageSize, FORWARD );
                        
			cout << "first result fft wKernel " << wKernel[0] << endl;
		}
		else
			
			// we're not using w-projection. just copy the AA-kernel into the w-kernel since this is
			// where we will look for it later.
			memcpy( wKernel, aaKernel, imageSize * imageSize * sizeof( complex<double> ) );
		
		// calculate separate kernels for each (oversampled) intermediate grid position.
		for ( int oversampleI = 0; oversampleI < _oversample; oversampleI++ )
			for ( int oversampleJ = 0; oversampleJ < _oversample; oversampleJ++ )
			{
				
				// get the index of the oversampled kernels.
				long int kernelIdx = (oversampleI * _kernelSize * _kernelSize) + (oversampleJ * _kernelSize * _kernelSize * _oversample);
						
				// add the index of the w-plane.
				kernelIdx = kernelIdx + kernelOffsetW;

				// copy the kernel from the temporary workspace into the actual kernel.
				updateKernel( &_kernel[ kernelIdx ], wKernel, _support, imageSupport, oversampleI, oversampleJ );
				
				// normalise the kernel following the FFT.
				if (_wPlanes > 1)
					for ( int i = 0; i < _kernelSize * _kernelSize; i++ )
						_kernel[ kernelIdx + i ] = _kernel[ kernelIdx + i ] / (double)(imageSize * imageSize);
				
			}

	}
	
	// cleanup memory.
	free( wKernel );
	free( aaKernel );
	
	// return success/failure.
	return ok;
	
} // generateKernel

//
//	getParameters()
//
//	CJS: 07/08/2015
//
//	Load the following parameters from the parameter file gridder-params: uv cell size, uv grid size, # w-planes, oversample.
//

void getParameters()
{

	char params[80], line[1024], par[80];
 
	// Open the parameter file and get all lines.
	FILE *fr = fopen( "gridder-prototype-params", "rt" );
	while (fgets( line, 80, fr ) != NULL)
	{

		sscanf( line, "%s %s", par, params );
		if (strcmp( par, CELL_SIZE ) == 0)
			_cellSize = atof( params );
		else if (strcmp( par, PIXELS_UV ) == 0)
			_uvPixels = atoi( params );
		else if (strcmp( par, W_PLANES ) == 0)
			_wPlanes = atoi( params );
		else if (strcmp( par, OVERSAMPLE ) == 0)
			_oversample = atof( params );
		else if (strcmp( par, INPUT_RA ) == 0)
			_inputRA = atof( params );
		else if (strcmp( par, INPUT_DEC ) == 0)
			_inputDEC = atof( params );
		else if (strcmp( par, OUTPUT_RA ) == 0)
			_outputRA = atof( params );
		else if (strcmp( par, OUTPUT_DEC ) == 0)
			_outputDEC = atof( params );
            
	}
	fclose( fr );
	
} // getParameters

//
//	gridVisibilities()
//
//	CJS: 10/08/2015
//
//	Produce an image of gridding visibilities by convolving the complex visibilities with the kernel function.
//

bool gridVisibilities( complex<double> * pGrid, complex<double> * pVisibility, int pOversample, int pKernelSize, int pSupport,
			complex<double> * pKernel, int pWPlanes, VectorF * pSample, int pNumSamples, int pNumChannels )
{
	const double INTERVAL = 0.05; // progress is updated at these % intervals.
	bool ok = true;
	
	// loop through the list of visilibities.
	// use the w-coordinate to pick an appropriate w-plane
	// use the u and v coordinates to pick an appropriate oversampled kernel.
	// loop through the kernel pixels, multiplying each one by the complex visibility and then adding to the grid.
	
	// clear the memory.
	memset( pGrid, 0, _uvPixels * _uvPixels * sizeof( complex<double> ) );
	
        
	// loop through the samples.
	int fraction = -1;
	for ( long int sample = 0; sample < pNumSamples; sample++ )
	{
		
		// display progress.
		if ((double)sample / (double)pNumSamples >= ((double)(fraction + 1) * INTERVAL))
		{
			fraction = fraction + 1;
#ifndef TESTING
			printf( "%i%%.", (int)(fraction * INTERVAL * 100) );
			fflush( stdout );
#endif
                }
			
                                        
		
		// loop through the channels.
		for ( long int channel = 0; channel < pNumChannels; channel++ )
		{
			
			// get the sample.
			VectorF uvwSample = pSample[ sample ];
			
			// convert uvw coordinates to units of lambda.
			uvwSample.u = uvwSample.u / _wavelength[ channel ];
			uvwSample.v = uvwSample.v / _wavelength[ channel ];
			uvwSample.w = uvwSample.w / _wavelength[ channel ];
                        
			// get the u index (plus remainer).
			double uExact = uvwSample.u / _uvCellSize;
			int uGrid = floor( uExact );
			int uOversample = (int)((uExact - (double)uGrid) * pOversample);
			uGrid = uGrid + (_uvPixels / 2.0);
			
			// get the v index (plus remained).
			double vExact = uvwSample.v / _uvCellSize;
			int vGrid = floor( vExact );
			int vOversample = (int)((vExact - (double)vGrid) * pOversample);
			vGrid = vGrid + (_uvPixels / 2.0);
				
			// identify which w-plane we should use for this sample.
			int wGrid = 0;
			if (pWPlanes > 1)
				for ( int i = pWPlanes - 1; i >= 0; i-- )
					if (uvwSample.w <= _wPlaneMax[ i ])
						wGrid = i;
		
			// calculate the kernel offset using the uOversample, vOversample and wGrid.
			long int kernelIdx = (uOversample * pKernelSize * pKernelSize) + (vOversample * pKernelSize * pKernelSize * pOversample);
						
			// add the index of the w-plane.
			kernelIdx = kernelIdx + (wGrid * pKernelSize * pKernelSize * pOversample * pOversample);
			
			// get the complex visibility.
			complex<double> visibility = pVisibility[ (sample * _numChannels) + channel ];
				
			// loop over the kernel in the u and v directions.
			for ( int i = 0; i < pKernelSize; i++ )
				for (int j = 0; j < pKernelSize; j++ )
				{
					
					// get exact grid coordinates,
					int iGrid = uGrid + i - pSupport;
					int jGrid = vGrid + j - pSupport;
					
					// get kernel value.
					complex<double> kernel = pKernel[ kernelIdx + (j * pKernelSize) + i ];
					
					// is this pixel within the grid range?
					if ((iGrid >= 0) && (iGrid < _uvPixels) && (jGrid >= 0) && (jGrid < _uvPixels))
					{
						
						// get pointer to grid.
						complex<double> * grid = &pGrid[ (jGrid * _uvPixels) + iGrid ];
						
						// update the grid.
						*grid = *grid + (visibility * kernel);
						
					}
			
				}
				
		}
		
	}
#ifndef TESTING
	printf("100%%\n\n");
#endif
	
	// return success/failure.
	return ok;
	
} // gridVisibilities

//
//	loadData()
//
//	CJS: 07/08/2015
//
//	Load the data from the measurement set. We need to load a list of sample (the uvw coordinates), a list of channels (we need the frequency of each channel),
//	and a list of visibility (we should have one visibility for each sample/channel combination).
//

bool loadData( char * pInputUVWFilename, char * pInputVisFilename, char * pInputChannelFilename )
{
	
	bool ok = true;
	char input_a[50], input_b[50], input_c[50];
	char line[1024];

	// read lines from input channel file to count the number of channels.
	_numChannels = 0;
	FILE * channels = fopen( pInputChannelFilename, "rt" );
	while ( fgets( line, 1024, channels ) != NULL )
		_numChannels = _numChannels + 1;
	fclose( channels );

	// reserve memory for wavelengths.
	_wavelength = (double *) malloc( sizeof( double ) * _numChannels );

	// read lines from input channels file again, storing wavelengths.
	int channel = 0;
	channels = fopen( pInputChannelFilename, "rt" );
	while ( fgets( line, 1024, channels ) != NULL )
	{
		sscanf(line, "%s", input_a);
		_wavelength[ channel ] = (double)(CONST_C / atof( input_a ));
		channel = channel + 1;
	}
	fclose( channels );

	// read lines from input file to count the number of samples.
	_numSamples = 0;
	FILE * input = fopen( pInputUVWFilename, "rt" );
	while ( fgets(line, 1024, input) != NULL )
		_numSamples = _numSamples + 1;
	fclose(input);
	
	// reserve memory for visibilities.
	_sample = (VectorF *) malloc( sizeof( VectorF ) * _numSamples );
	
	// read lines from input file again, storing visibilies.
	int sample = 0;
	input = fopen( pInputUVWFilename, "rt" );
	while ( fgets(line, 1024, input) != NULL )
	{
		sscanf(line, "%s %s %s", input_a, input_b, input_c);
		_sample[sample].u = atof( input_a );
		_sample[sample].v = atof( input_b );
		_sample[sample].w = atof( input_c );
		sample = sample + 1;
	}
	fclose(input);
	
	// reserve memory for visibilities.
	_visibility = (complex<double> *) malloc( sizeof( complex<double> ) * _numSamples * _numChannels );
	
	// read lines from input file again, storing visibilies.
	int visibility = 0;        
	input = fopen( pInputVisFilename, "rt" );
	while ( fgets(line, 1024, input) != NULL )
	{
		sscanf(line, "%s %s", input_a, input_b);
		_visibility[visibility] = complex<double>( atof( input_a ), atof( input_b ) );
		visibility = visibility + 1;
	}
	fclose(input);
	
	// return success/failure.
	return ok;
	
} // loadData

//
//	saveBitmap()
//
//	CJS: 10/08/2015
//
//	Save the current grid as a bitmap file.
//

bool saveBitmap( char * pOutputFilename, complex<double> * pGrid )
{
	
	unsigned char * image = NULL;
	
	const int HEADER_SIZE = 1078;
	
	// allocate and build the header.
	unsigned char * fileHeader = (unsigned char *) malloc( HEADER_SIZE );
	memset( fileHeader, 0, HEADER_SIZE );

	// file header.
	fileHeader[BIT_CONST] = 'B'; fileHeader[MAP_CONST] = 'M';					// bfType
	int size = (_uvPixels * _uvPixels) + HEADER_SIZE; memcpy( &fileHeader[IMAGE_SIZE], &size, 4 );	// bfSize
	int offBits = HEADER_SIZE; memcpy( &fileHeader[FILE_HEADER_SIZE], &offBits, 4 );		// bfOffBits

	// image header.
	size = 40; memcpy( &fileHeader[BITMAP_INFO_HEADER], &size, 4 );					// biSize
	memcpy( &fileHeader[IMAGE_WIDTH], &_uvPixels, 4 );						// biWidth
	memcpy( &fileHeader[IMAGE_HEIGHT], &_uvPixels, 4 );						// biHeight
	short planes = 1; memcpy( &fileHeader[COLOUR_PLANES], &planes, 2 );				// biPlanes
	short bitCount = 8; memcpy( &fileHeader[BIT_COUNT], &bitCount, 2 );				// biBitCount
	int coloursUsed = 256; memcpy( &fileHeader[COLOURS_USED], &coloursUsed, 4 );			// biClrUsed

	// colour table.
	for (unsigned int i = 0; i < 256; ++i)
	{
		unsigned int colour = (i << 16) + (i << 8) + i;
		memcpy( &fileHeader[54 + (i * 4)], &colour, 4 );
	}
	
	bool ok = true;

	// open file.
	FILE * outputFile = fopen( pOutputFilename, "w" );
	if (outputFile == NULL)
	{
		printf( "Could not open file \"%s\".\n", pOutputFilename );
		ok = false;
	}
	else
	{

		// write the file header.
		size_t num_written = fwrite( fileHeader, 1, 1078, outputFile );
		if (num_written != 1078)
		{
			printf( "Error: cannot write to file.\n" );
			ok = false;
		}
		
		// find the maximum and minimum pixel values.
		double min = abs( pGrid[0] );
		double max = abs( pGrid[0] );
		for ( int i = 0; i < _uvPixels * _uvPixels; i++ )
		{
			if (abs( pGrid[i] ) < min)
				min = abs( pGrid[i] );
			if (abs( pGrid[i] ) > max)
				max = abs( pGrid[i] );
		}
#ifndef TESTING
		printf("min - %f, max - %f\n", min, max );
#endif
		// add 1% allowance to max - we don't want saturation.
		max = ((max - min) * 1.01) + min;
		
		// construct the image.
		image = (unsigned char *) malloc( _uvPixels * _uvPixels * sizeof( unsigned char ) );
		for ( int i = 0; i < _uvPixels * _uvPixels; i++ )
			image[i] = (unsigned char)( (abs( pGrid[i] ) - min) * ((double)256 / max) );
		
		// write the data.
		if (ok == true)
		{
			
			size_t num_written = fwrite( image, 1, _uvPixels * _uvPixels, outputFile );
			if (num_written != (_uvPixels * _uvPixels))
			{
				printf( "Error: cannot write to file.\n" );
				ok = false;
			}
			
		}

		// close file.
		fclose( outputFile );
		
	}

	// cleanup memory.
	free( (void *) fileHeader );
	if (image != NULL)
		free( image );
	
	// return success flag.
	return ok;
	
} // saveBitmap

//
//	generateImageOfConvolutionFunction()
//
//	CJS: 05/01/2016
//
//	Generate the deconvolution function by gridding a single source at u = 0, v = 0, and FFT'ing.
//

bool generateImageOfConvolutionFunction( char * pDeconvolutionFilename )
{
	
	bool ok = true;
	
	// create the deconvolution image.
	_deconvolutionImage = (complex<double> *) malloc( _uvPixels * _uvPixels * sizeof( complex<double> ) );
        
	// create a single visibility with a value of 1.
	complex<double> * tmpVisibility = (complex<double> *) malloc( sizeof( complex<double> ) );
	tmpVisibility[ 0 ] = 1;
        
	// create a single uvw vector with a value of <0, 0, 0>.
	VectorF * tmpSample = (VectorF *) malloc( sizeof( VectorF ) );
	tmpSample[ 0 ].u = 0;
	tmpSample[ 0 ].v = 0;
	tmpSample[ 0 ].w = 0;
		
	// generate the deconvolution function by gridding a single visibility.
#ifndef TESTING
	printf( "\ngridding visibilities for deconvolution function.....\n" );
#endif
        
	ok = gridVisibilities( _deconvolutionImage, tmpVisibility, 1, _aaKernelSize, _aaSupport, _aaKernel, 1, tmpSample, 1, 1 );
        
	// FFT the gridded data to get the deconvolution map.
	performFFT( _deconvolutionImage, _uvPixels, INVERSE );
	
	// clean up memory.
	free( tmpVisibility );
	free( tmpSample );
	
	// save the deconvolution image.
#ifndef TESTING
	printf( "\npixel intensity range of deconvolution image:\n" );
#endif
	ok = saveBitmap( pDeconvolutionFilename, _deconvolutionImage );
	
	// return success flag.
	return ok;
	
} // generateImageOfConvolutionFunction


#ifndef TESTING
//
//	main()
//
//	CJS: 07/08/2015
//
//	Main processing.
//

int main( int pArgc, char ** pArgv )
{
	
	char DIRTY_BEAM_EXTENSION[] = "-dirty-beam.bmp";
	char CLEAN_BEAM_EXTENSION[] = "-clean-beam.bmp";
	char GRIDDED_EXTENSION[] = "-gridded.bmp";
	char DIRTY_IMAGE_EXTENSION[] = "-dirty-image.bmp";
	char CLEAN_IMAGE_EXTENSION[] = "-clean-image.bmp";
	char RESIDUAL_IMAGE_EXTENSION[] = "-residual-image.bmp";
	char DECONVOLUTION_EXTENSION[] = "-deconvolution.bmp";
	char UVW_PREFIX[] = "uvw-";
	char DATA_PREFIX[] = "data-";
	char CHANNEL_PREFIX[] = "channel-";
	char INPUT_EXTENSION[] = ".txt";
	
	char outputGriddedFilename[ 100 ];
	char outputDirtyImageFilename[ 100 ];
	char outputDeconvolutionFilename[ 100 ];
	
	struct timespec time1, time2;
	
	// read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
	if (pArgc != 3)
	{
		printf("Wrong number of arguments. Require the input input identifier and bitmap prefix.\n");
		return 1;
	}
	
	char inputUVWFilename[100];
	char inputVisFilename[100];
	char inputChannelFilename[100];
	char * inputIdentifier = pArgv[1];
	char * filenamePrefix = pArgv[2];
	
	// build input filenames.
	strcpy( inputUVWFilename, UVW_PREFIX ); strcat( inputUVWFilename, inputIdentifier ); strcat( inputUVWFilename, INPUT_EXTENSION );
	strcpy( inputVisFilename, DATA_PREFIX ); strcat( inputVisFilename, inputIdentifier ), strcat( inputVisFilename, INPUT_EXTENSION );
	strcpy( inputChannelFilename, CHANNEL_PREFIX ); strcat( inputChannelFilename, inputIdentifier ), strcat( inputChannelFilename, INPUT_EXTENSION );
	
	// build output filenames.
	strcpy( outputGriddedFilename, filenamePrefix ); strcat( outputGriddedFilename, GRIDDED_EXTENSION );
	strcpy( outputDirtyImageFilename, filenamePrefix ); strcat( outputDirtyImageFilename, DIRTY_IMAGE_EXTENSION );
	strcpy( outputDeconvolutionFilename, filenamePrefix ); strcat( outputDeconvolutionFilename, DECONVOLUTION_EXTENSION );

	// get the parameters from file.
	getParameters();

	clock_gettime( CLOCK_REALTIME, &time1 );

	// load data. we need to load 'samples' (the uvw coordinates), 'channels' (frequency of each channel) and 'visibilities' (one for each sample/channel combination).
	bool ok = loadData( inputUVWFilename, inputVisFilename, inputChannelFilename );
	if (ok == true)
	{

		clock_gettime( CLOCK_REALTIME, &time2 );
		fprintf( stderr, "\n--- time (load data): (%f ms) ---\n\n", getTime( time1, time2 ) );
	
		// calculate kernel size and other parameters.
		calculateKernelSize();

		clock_gettime( CLOCK_REALTIME, &time1 );
		fprintf( stderr, "\n--- time (calculate kernel size): (%f ms) ---\n\n", getTime( time2, time1 ) );
	
		// generate kernel.
		ok = generateKernel();

		clock_gettime( CLOCK_REALTIME, &time2 );
		fprintf( stderr, "--- time (generate kernel): (%f ms) ---\n", getTime( time1, time2 ) );

	}
		
	// do phase correction.
	doPhaseCorrection();

	clock_gettime( CLOCK_REALTIME, &time1 );
	fprintf( stderr, "\n--- time (phase rotation): (%f ms) ---\n", getTime( time2, time1 ) );
	
	if (ok == true)
	{
		
		// generate image of convolution function.
		ok = generateImageOfConvolutionFunction( outputDeconvolutionFilename );

		clock_gettime( CLOCK_REALTIME, &time2 );
		fprintf( stderr, "\n--- time (generate deconvolution map): (%f ms) ---\n", getTime( time1, time2 ) );
		
	}
	
	if (ok == true)
	{
	
		// create the grid.
		_grid = (complex<double> *) malloc( _uvPixels * _uvPixels * sizeof( complex<double> ) );
	
		// grid visibilities.
		printf( "\ngridding visibilities for dirty image.....\n" );
		ok = gridVisibilities( _grid, _visibility, _oversample, _kernelSize, _support, _kernel, _wPlanes, _sample, _numSamples, _numChannels );

		clock_gettime( CLOCK_REALTIME, &time1 );
		fprintf( stderr, "--- time (gridding for dirty image): (%f ms) ---\n\n", getTime( time2, time1 ) );
		
	}
	
	if (ok == true)
	{
	
		// save image.
		printf( "pixel intensity range of uv data:\n" );
		ok = saveBitmap( outputGriddedFilename, _grid );

		clock_gettime( CLOCK_REALTIME, &time2 );
		fprintf( stderr, "\n--- time (save gridded image): (%f ms) ---\n\n", getTime( time1, time2 ) );
	
		// make dirty image.
		performFFT( _grid, _uvPixels, INVERSE );

		clock_gettime( CLOCK_REALTIME, &time1 );
		fprintf( stderr, "\n--- time (fft): (%f ms) ---\n", getTime( time2, time1 ) );
				
		// normalise the image following the FFT.
		for ( int i = 0; i < _uvPixels * _uvPixels; i++ )
			_grid[ i ] = _grid[ i ] / (double)(_uvPixels * _uvPixels);
		
		// divide dirty image by deconvolution image.
		double maxValue = -1;
		for ( int i = 0; i < _uvPixels; i++ )
			for ( int j = 0; j < _uvPixels; j++ )
			{
				if (abs( _deconvolutionImage[ (j * _uvPixels) + i ] ) == 0)
					_grid[ (j * _uvPixels) + i ] = 0;
				else
					_grid[ (j * _uvPixels) + i ] = _grid[ (j * _uvPixels) + i ] / abs( _deconvolutionImage[ (j * _uvPixels) + i ] );
				if (abs( _grid[ (j * _uvPixels) + i ] ) > maxValue)
					maxValue = abs( _grid[ (j * _uvPixels) + i ] );
			}
		printf( "deconvolved image max value: %f\n", maxValue );
	
		// save dirty image.
		printf( "\npixel intensity range of dirty image:\n" );
		ok = saveBitmap( outputDirtyImageFilename, _grid );

		clock_gettime( CLOCK_REALTIME, &time2 );
		fprintf( stderr, "\n--- time (save dirty image): (%f ms) ---\n", getTime( time1, time2 ) );
		
	}
	
	// clear memory.
	if (_sample != NULL)
		free( _sample );
	if (_wavelength != NULL)
		free( _wavelength );
	if (_visibility != NULL)
		free( _visibility );
	if (_kernel != NULL)
		free( _kernel );
	if (_grid != NULL)
		free( _grid );
	if (_deconvolutionImage != NULL)
		free( _deconvolutionImage );
	if (_aaKernel != NULL)
		free( _aaKernel );
	if (_wPlaneMean != NULL)
		free( _wPlaneMean );
	if (_wPlaneMax != NULL)
		free( _wPlaneMax );
	
	return true;
	
} // main
#endif
