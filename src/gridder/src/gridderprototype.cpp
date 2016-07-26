#include "gridderprototype.h"
#include "phasecorrection.h"

using namespace std;
using namespace arma;

#define INTERVAL 0.05 // progress is updated at these % intervals.
#define HEADER_SIZE 1078

int inverse = INVERSE;
int forward = FORWARD;

//
//	GLOBAL VARIABLES
//

// grid parameters.
double _cellSize = 0.0;
double _uvCellSize = 0.0;
int _uvPixels = 0;
int _wPlanes = 0;
double _fixedFrequency = 0.0;
double _inputRA = 0.0;
double _inputDEC = 0.0;
double _outputRA = 0.0;
double _outputDEC = 0.0;
int _numSamples = 0;
mat _sample;
int _numChannels = 0;
vec _wavelength;
rowvec _frequency;
vec _wPlaneMean;
vec _wPlaneMax;
cx_rowvec _visibility;
double _oversample = 0.0;
int _support = 0;
int _kernelSize = 0;
int _aaSupport = 0;
int _aaKernelSize = 0;
int _wSupport = 0;
int _wKernelSize = 0;
cx_rowvec _kernel;
cx_rowvec _grid;
cx_vec _deconvolutionImage;
cx_rowvec _aaKernel;


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

        int num_samp_chan = _numSamples * _numChannels;
        vec wvalue_vector(num_samp_chan);


        // calculate the size of each uv pixel.
        _uvCellSize =  double(1/(DEG_TO_RAD((_cellSize / 3600.0))*_uvPixels));

        // if we are using w-projection, we need to determine the distribution of w-planes.
        if (_wPlanes > 1)
        {
            for(int i=0; i<_numSamples; i++) {
                for(int j=0; j<_numChannels; j++) {
                    wvalue_vector[ (i * _numChannels) + j ] =_sample.at(i,2) / _wavelength[ j ];
               }
           }


                // sort these w-values.
                wvalue_vector = sort(wvalue_vector);

                // divide the w-values up into the required number of planes. the planes are therefore not
                // evenly spaced, but spaced to match the density distribution of w-values.
               int visibilitiesP_per_wplane = num_samp_chan / _wPlanes;


                _wPlaneMax.resize(_wPlanes);
                _wPlaneMean.resize(_wPlanes);

                int lower_index(0), upper_index(0);
                int i(0);


                for(i =0;  i < _wPlanes; i++)
                {
                    lower_index = i * visibilitiesP_per_wplane;
                    upper_index= lower_index + visibilitiesP_per_wplane-1;

                    if (i == (_wPlanes - 1)){
                       upper_index = num_samp_chan-1;
                       _wPlaneMax[i] = wvalue_vector[upper_index];
                       _wPlaneMean[i] = mean(wvalue_vector( span(lower_index, upper_index) ));
                    }
                    else {
                        _wPlaneMax[i] = wvalue_vector[upper_index];
                        _wPlaneMean[i] = mean(wvalue_vector( span(lower_index, upper_index) ));
                    }
                }
    }


           cout << "Maximum w-value: " << _wPlaneMax[ _wPlanes - 1 ] << " lambda" << endl;
         //   cout << "Mean  w-value: " << _wPlaneMean<< endl;

           // set the properties of the anti-aliasing kernel.
           _aaSupport = 3;
           _aaKernelSize = (2 * _aaSupport) + 3;

           // set the properties of the w-kernel.
           _wSupport = 0;
           if (_wPlanes > 1) {
               _wSupport = 25;
           }
           _wKernelSize = (2 * _wSupport) + 1;

           // calculate the support, which should be either the w-support or the aa-support, whichever is larger.
           if (_aaSupport > _wSupport) {
               _support = _aaSupport;
           }
           else {
               _support = _wSupport;
           }
           // calculate the kernel size. thix would normally be 2.S + 1, but I add an extra couple of cells in order that
           // my implementation of oversampling works correctly.
           _kernelSize = (2 * _support) + 3;

            cout << "_support = " << _support << endl;
            cout << "_aaSupport = " << _aaSupport << endl;
            cout << "_wSupport = " << _wSupport << endl;
            cout << "_cellSize = " << _cellSize << " arcsec = " << DEG_TO_RAD(_cellSize / 3600) << endl;
            cout << "_uvCellSize = " << _uvCellSize << endl;


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

        PhaseCorrection phaseCorrection;

        // set up the coordinate system.
        phaseCorrection.inCoords.longitude = _inputRA;
        phaseCorrection.inCoords.latitude = _inputDEC;
        strcpy( phaseCorrection.inCoords.epoch, "J2000" );

        phaseCorrection.outCoords.longitude = _outputRA;
        phaseCorrection.outCoords.latitude = _outputDEC;
        strcpy( phaseCorrection.outCoords.epoch, "J2000" );

        phaseCorrection.uvProjection = 0;

        // initialise phase rotation matrices.
        phaseCorrection.init();

        cx_double phasor = 0.0;

        // loop through all the visibilities.
        for ( int i = 0; i < _numSamples; i++ )
        {
            // get the rotated uvw coordinate.
            phaseCorrection.uvwIn[0] = _sample.at(i,0);
            phaseCorrection.uvwIn[1]= _sample.at(i,1);
            phaseCorrection.uvwIn[2] = _sample.at(i,2);
            phaseCorrection.rotate();
            _sample.at(i,0) = phaseCorrection.uvwOut[0];
            _sample.at(i,1)  = phaseCorrection.uvwOut[1];
            _sample.at(i,2)  = phaseCorrection.uvwOut[2];

            // loop through all the channels.
            for ( int j = 0; j < _numChannels; j++ ) {
               phasor = cx_double( cos( 2 * M_PI * phaseCorrection.phase / _wavelength[ j ] ),
                        
               // multiply phasor by visibility.
               _visibility[ (i * _numChannels) + j ] *= phasor;
            }
        }




} // doPhaseCorrection


void updateKernel( cx_rowvec &pKernel, cx_rowvec &pImage, int pKernelSupport, int pImageSupport, int pOversampleI, int pOversampleJ, long startIndex )
{

    int kernel_size((pKernelSupport * 2) + 3);
    int image_size((pImageSupport * 2) + 1);
    int oversampled_kernel_size(((pKernelSupport * 2) + 1) * _oversample);
    int oversampled_kernel_support(lrint(floor( (oversampled_kernel_size - 1) / 2.0) ));
    int start(-lrint(floor( round(_oversample) / 2.0 )));
    cx_double total(0.0);
    int i(0);
    int j(0);
    vec image(kernel_size+_oversample-1);

    while( i < kernel_size){
        while( j <_oversample) {
             image[ i ] =  start + (i * _oversample) - pOversampleJ + pImageSupport - oversampled_kernel_support + j;
             j++;
             i++;
         }
         image[ i ] =  start + (i * _oversample) - pOversampleJ + pImageSupport - oversampled_kernel_support;
         i++;
    }


    for( j=0; j< kernel_size; j++) {
      for( i=0; i<kernel_size; i++)
      {
          total += pImage[ (image[ i ] * image_size) + image[ j ] ];
          pKernel[ (i * kernel_size) + j + startIndex] = total / (round(_oversample*_oversample) );
          total = cx_double( 0, 0 );
      }
     }


} // updateKernel



//
//	generateWKernel()
//
//	CJS: 29/04/2016
//
//	Generate the W-kernel.
//

void generateWKernel( cx_rowvec &pWKernel, int pImageSize, double pW, double pcell_size_radians )
{


    // calculate the image support. this will be an integer since the image size is odd.
    int image_support((pImageSize - 1) / 2);

    // calculate the cell size squared.
    double cell_size_radians_squared(pcell_size_radians * pcell_size_radians);
    double x(0.0);
    double y(0.0);
    double r_squared_radians(0.0);

    for ( int i = 0; i < pImageSize; i++ )
     {
            // calculate the x offset from the centre of the image.
            x =  ( (round(i) - image_support) * (round(i) - image_support) );

            for ( int j = 0; j < pImageSize; j++ )
            {

                // calculate the y offset from the centre of the image.
                y = (round(j) - image_support) * (round(j) - image_support);

                // calculate r^2 (dist from centre of image squared).
                // convert r^2 from pixels to radians.
                r_squared_radians = (x + y) * cell_size_radians_squared;

                // scale r^2 by the oversampling parameter O, we want to shrink our function by a factor of O so that
                // the kernel in the uv domains appears O times larger.
                r_squared_radians *= round(_oversample * _oversample);

                // calculate kernel values,
                pWKernel( (j * pImageSize) + i ) = cx_double( (cos( 2 * M_PI * pW * (sqrt( 1.0 - r_squared_radians ) - 1.0) )/ sqrt( 1.0 - r_squared_radians )) , ( sin( 2 * M_PI * pW * (sqrt( 1.0 - r_squared_radians ) - 1.0) )/ sqrt( 1.0 - r_squared_radians )) );

            }

        }


} // generateWKernel






void generateAAKernel( cx_rowvec &pAAKernel, int pOversampledKernelSize, int pImageSize )
{

    int const NP = 4;
    int const NQ =  2;


     // data for spheroidal gridding function.
     mat dataP =  {	{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                    {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}
                  };

     mat dataQ = { {1.0, 8.212018e-1, 2.078043e-1},
                   {1.0, 9.599102e-1, 2.918724e-1}
                 };

    // calculate the support of the image and the oversampled kernel. we use an integer for the image support because
    // we know the image size is odd, which implies an integer support size.
    double y(0.0);
    double val(0.0);
    double spheroid_radius(0.0);
    double radius_end(0.0);
    double top(0.0);
    int part(0);
    double del_radius_sq(0.0);
    double bottom(0.0);
    double x(0.0);
    double r(0.0);
    int image_i(0);
    int image_j(0);


    // calculate the support of the image and the oversampled kernel. we use an integer for the image support because
    // we know the image size is odd, which implies an integer support size.
    double oversampled_support( ( round(pOversampledKernelSize) - 1.0) / 2.0 );
    int image_support( ( round(pImageSize) - 1.0) / 2.0 );

    // now loop through the i and j (u and v) coordinates of the kernel.
    for ( int i = 0; i < pOversampledKernelSize; i++ )
    {
        image_i = i + image_support - lrint(ceil( oversampled_support )); // get the i coordinates for the whole image.
        x = round(i) - oversampled_support; // calculate x-offset from centre of kernel.

            for ( int j = 0; j < pOversampledKernelSize; j++ )
            {
                // get the j coordinates for the whole image.
                image_j = j + image_support - lrint(ceil( oversampled_support )); 

                // calculate y-offset from centre of kernel/image.    
                y = round(j) - oversampled_support; 
                
                // calculate r, which is the offset from the centre of the kernel in dimensional grid units.
                r = sqrt( (x * x) + (y * y) ); 
                
                // get the distance (0 to 1) from the centre of the kernel to the first minimum of the prolate spheroidal.
                spheroid_radius = r / round(oversampled_support); 

                // now, calculate the anti-aliasing kernel.
                // only calculate a kernel value if this pixel is within the required radius from the centre.
                val =0.0;

                // only calculate a kernel value if this pixel is within the required radius from the centre.
                if (spheroid_radius <= 1)
                {
                    part = 1;
                    radius_end = 1.0;
                    
                    if (spheroid_radius < 0.75)
                    {
                       part = 0;
                       radius_end = 0.75;
                    }

                    del_radius_sq = (spheroid_radius * spheroid_radius) - (radius_end * radius_end);

                    top=0.0;
                    for (int k = 0; k < NP; k++) {
                        top += (dataP.at( part, k) * pow( del_radius_sq, k ));
                    }

                    bottom = 0.0;
                    for (int k = 0; k < NQ; k++) {
                         bottom += (dataQ.at(part, k) * pow( del_radius_sq, k ));
                    }

                    if (bottom != 0) {
                         val = top / bottom;
                    }

                    // the gridding function is (1 - spheroid_radius^2) x gridsf
                    val *= (1 - (spheroid_radius * spheroid_radius));

                }

                // update the appropriate pixel in the anti-aliasing kernel.
                pAAKernel[ (image_j * pImageSize) + image_i ]= cx_double( val, 0 );

            }

    }

}


//
//	generateKernel()
//
//	CJS: 16/12/2015
//
//	Generates the convolution function. A separate kernel is generated for each w-plane, and for each oversampled intermediate
//	grid position.
//

int generateKernel()
{

        int ok(1);
        double cell_size_radians(DEG_TO_RAD(_cellSize / 3600.0));
        long  kernel_idx(0);


        // calculate the size of the images we want to use to construct our kernels. we choose to make this number odd, so
        // we will use the global grid size if it is odd, otherwise we use grid size - 1 pixel.
        int image_size(_uvPixels);

        if ( (image_size / 2.0) == floor( image_size / 2.0 )) {
                --image_size;
        }


        int image_support((image_size - 1) / 2); // image support. this is the centre of the image, and since image size is odd then the support will be an integer.

        _kernel.resize(_kernelSize * _kernelSize * _oversample * _oversample * _wPlanes); // reserve some memory for the kernels.
        _aaKernel.resize( _aaKernelSize * _aaKernelSize );  // reserve some memory for the global anti-aliasing kernel. this is used to remove the kernel convolution.

        cx_rowvec w_kernel(  image_size * image_size); // reserve some memory for a w-kernel.
        cx_rowvec aa_kernel( image_size * image_size ); // reserve some memory for a aa-kernel.

        // calculate the oversampled AA kernel size. if this figure is even, subtract one. we MUST have an odd kernel size
        // in order that our kernel function, which is positioned on the centre of a pixel rather than between pixels, is centred
        // within the kernel.
        int oversampledAAKernelSize(((_aaSupport * 2) + 1) * _oversample);

        if ( (oversampledAAKernelSize / 2.0) == floor( oversampledAAKernelSize / 2.0 )) {
            oversampledAAKernelSize--;
        }

        // generate the anti-alliasing kernel in image [aaKernel] of size [image_size].
        // the kernel will be prolate spheroidal function of width [oversampledKernelSize]
        generateAAKernel( aa_kernel, oversampledAAKernelSize, image_size );

        updateKernel( _aaKernel, aa_kernel, _aaSupport, image_support, 0, 0, 0);


        if (_wPlanes > 1) {
          performFFT( aa_kernel, image_size, INVERSE );
        }

        long  kernelOffsetW(0);
        
        // calculate separate kernels for each w-value.
        for ( int w_plane = 0; w_plane < _wPlanes; w_plane++ )
        {
            // calculate kernel offset.
            kernelOffsetW = w_plane * _kernelSize * _kernelSize * _oversample * _oversample;

            if (_wPlanes > 1)
            {
                // generate a w-kernel with size [image_size] in [w_kernel].
                generateWKernel( w_kernel, image_size, _wPlaneMean[ w_plane ], cell_size_radians );
    
                // now we need to convolve the w-kernel and the AA-kernel. we don't do this is are not
                // using w-projection. we multiply the kernels in the image domain.
                w_kernel = conv2(w_kernel, aa_kernel);
             }
             else {
                w_kernel = aa_kernel;
             }

             kernel_idx = 0;
             // calculate separate kernels for each (oversampled) intermediate grid position.
             for ( int oversampleI = 0; oversampleI < _oversample; oversampleI++ )
                for ( int oversampleJ = 0; oversampleJ < _oversample; oversampleJ++ )
                {
                    // add the index of the w-plane.
                    kernel_idx = (oversampleI * _kernelSize * _kernelSize) + (oversampleJ * _kernelSize * _kernelSize * _oversample) + kernelOffsetW;

                    // copy the kernel from the temporary workspace into the actual kernel.
                    updateKernel( _kernel, w_kernel, _support, image_support, oversampleI, oversampleJ, kernel_idx);

                    // normalise the kernel following the FFT.
                    if (_wPlanes > 1) {
                        for ( int i = 0; i < _kernelSize * _kernelSize; i++ ) {
                            _kernel[ kernel_idx + i ] /= round(image_size * image_size);
                        }
                    }
                } 
        }

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
    FILE *fr = fopen( "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/gridder-prototype-params", "rt" );
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
        else if (strcmp( par, FREQUENCY ) == 0)
            _fixedFrequency = atof( params );
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

int gridVisibilities( cx_rowvec &pGrid, cx_rowvec pVisibility, int pOversample, int pKernelSize, int pSupport,
                        cx_vec pKernel, int pWPlanes, mat pSample, int pNumSamples, int pNumChannels )
{

    int ok(1);

    // loop through the list of visilibities.
    // use the w-coordinate to pick an appropriate w-plane
    // use the u and v coordinates to pick an appropriate oversampled kernel.
    // loop through the kernel pixels, multiplying each one by the complex visibility and then adding to the grid.

    // loop through the samples.
    int fraction(-1);
    int u_oversample(0);
    int u_grid(0);
    int v_oversample(0);
    int v_grid(0);
    int w_grid(0);
    int i_grid(0);
    int j_grid(0);
    double u_exact(0.0);
    double v_exact(0.0);
    double u_fract_part(0.0);
    double u_int_part(0.0);
    double v_int_part(0.0);
    double v_fract_part(0.0);
    vec3 uvw_sample;
    cx_double visibility, kernel;
    long  kernel_idx(0);
    int ind   x(0);

    for ( int sample = 0; sample < pNumSamples; sample++ )
    {
        // loop through the channels.
        for ( int channel = 0; channel < pNumChannels; channel++ )
        {
            // convert uvw coordinates to units of lambda.
            uvw_sample[0] /= _wavelength[ channel ];
            uvw_sample[1] /=  _wavelength[ channel ];
            uvw_sample[2] /=  _wavelength[ channel ];

            // get the u index (plus remainer).
            u_exact = uvw_sample[0] / _uvCellSize;
            v_exact = uvw_sample[1] / _uvCellSize;
            u_fract_part = modf (uvw_sample[0] / _uvCellSize , &u_int_part);
            v_fract_part = modf (v_exact , &v_int_part);
            
            // get the u and v index (plus remained).
            u_oversample = u_fract_part * pOversample;
            u_grid = u_int_part + (_uvPixels / 2.0);
            v_oversample = v_fract_part * pOversample;
            v_grid = v_int_part + (_uvPixels / 2.0);

            // identify which w-plane we should use for this sample.
            w_grid = 0;

            if (pWPlanes > 1)
            {
                for ( int i = (pWPlanes - 1); i >= 0; i-- ) {
                    if ( uvw_sample[3] <= _wPlaneMax[ i ] ){
                        w_grid = i;
                    }
                }
            }
            // calculate the kernel offset using the u_oversample, v_oversample and w_grid.
            // add the index of the w-plane.
            kernel_idx = (u_oversample * pKernelSize * pKernelSize) + (v_oversample * pKernelSize * pKernelSize * pOversample) + (w_grid * pKernelSize * pKernelSize) * pOversample * pOversample;

            // get the complex visibility.
            visibility = pVisibility[ (sample * _numChannels) + channel ];

            indx = 0;
            // loop over the kernel in the u and v directions.
            for ( int i = 0; i < pKernelSize; i++ ){
                for (int j = 0; j < pKernelSize; j++ )
                {
                    // get exact grid coordinates,
                    i_grid = u_grid + i - pSupport;
                    jgrid = v_grid + j - pSupport;

                    // get kernel value.
                    kernel = pKernel[ kernel_idx + (j * pKernelSize) + i ];

                    // is this pixel within the grid range?
                    if ((i_grid >= 0) && (i_grid < _uvPixels) && (j_grid >= 0) && (j_grid < _uvPixels))
                    {
                        indx = (j_grid * _uvPixels) + i_grid;
                        pGrid( indx ) = pGrid[ indx ] + (visibility * kernel);
                    }
                }
            }
        }
    }

    return ok;

} // gridVisibilities


//
//	performFFT()
//
//	CJS: 11/08/2015
//
//	Make a dirty image by inverse FFTing the gridded visibilites.
//
void performFFT( cx_rowvec &pGrid, int pSize, fftdirection pFFTDirection )
{

    cx_mat p_grid_mat(pSize, pSize);
    cx_mat fft_mat(pSize, pSize, fill::zeros);
    int pivot  = 0;
    int pivot2 =0;

    if(pSize % 2 != 0) {
        pivot = lrint(ceil(double(pSize) / 2.0));
        pivot2 = lrint(floor(double(pSize) / 2.0));
    }
    else {
        pivot = pSize/2;
    }

    int x =0;
    for(int row=0; row<pSize; row++) {
       for(int col=0; col<pSize; col++)
       {
            p_grid_mat(row,col) = pGrid[x];
            x++;
       }
    }

    cx_mat kernel_shifted = shift(p_grid_mat, -pivot, 1);
    kernel_shifted = shift(kernel_shifted, pivot2, 0);

    if(pFFTDirection == INVERSE) {
       fft_mat  = ifft2(kernel_shifted*(pSize*pSize));
    }
    else {
        fft_mat = fft2(p_grid_mat);
    }

    pGrid = vectorise(fft_mat,1);


} // performFFT


//
//	loadData()
//
//	CJS: 07/08/2015
//
//	Load the data from the measurement set. We need to load a list of sample (the uvw coordinates), a list of channels (we need the frequency of each channel),
//	and a list of visibility (we should have one visibility for each sample/channel combination).
//

int loadData( char * pInputUVWFilename, char * pInputVisFilename, char * pInputChannelFilename )
{

        int ok(1);
        char input_a[50], input_b[50], input_c[50];
        char line[1024];

        // set up 5 frequencies from 1 GHz to 5 GHz.
        numChannels = 0;
        FILE * channels = fopen( pInputChannelFilename, "rt" );
        
        while ( fgets( line, 1024, channels ) != NULL ){
                _numChannels++;
        }
        
        fclose( channels );

        _wavelength.resize(_numChannels);

        int channel = 0;
        channels = fopen( pInputChannelFilename, "rt" );

        while ( fgets( line, 1024, channels ) != NULL )
        {
            sscanf(line, "%s", input_a);
            _wavelength[ channel ] = double(CONST_C / atof( input_a ));
            channel++;
        }
        
        fclose( channels );

        // read lines from input file to count the number of samples.
        _numSamples = 0;
        FILE * input = fopen( pInputUVWFilename, "rt" );
        
        while ( fgets(line, 1024, input) != NULL ){
            _numSamples ++;
        }    
         
        fclose(input);

        
        // reserve memory for visibilities.
        _sample.resize( _numSamples, 3);
               
        // read lines from input file again, storing visibilies.
        int sample = 0;               
        input = fopen( pInputUVWFilename, "rt" );        
        while ( fgets(line, 1024, input) != NULL )
        {   
            sscanf(line, "%s %s %s", input_a, input_b, input_c);
            _sample(sample,0) = atof( input_a );
            _sample(sample,1)  = atof( input_b );
            _sample(sample,2)  = atof( input_c );
            sample++;
        }
            
        fclose(input);
        
        // reserve memory for visibilities.
        _visibility.resize(_numSamples * _numChannels );

        // read lines from input file again, storing visibilies.       
        int visibility = 0;
        input = fopen( pInputVisFilename, "rt" );
        while ( fgets(line, 1024, input) != NULL )
        {
            sscanf(line, "%s %s", input_a, input_b);
            _visibility(visibility) = cx_double( atof( input_a ), atof( input_b ) );
            visibility++; 
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

int saveBitmap( char * pOutputFilename, cx_rowvec &pGrid )
{
    
    unsigned char * image = NULL;
    
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
    
    int ok = 1;
    
    // open file.
    FILE * outputFile = fopen( pOutputFilename, "w" );
    if (outputFile == NULL)
    {
        printf( "Could not open file \"%s\".\n", pOutputFilename );
        ok = 0;
    }
    else
    {
        // write the file header.
        size_t num_written = fwrite( fileHeader, 1, 1078, outputFile );
        if (num_written != 1078)
        {
            printf( "Error: cannot write to file.\n" );
            ok = 0;
        }
        
        double min = abs( pGrid[0] );
        double max = abs( pGrid[0] );
        
        for ( int i = 0; i < _uvPixels * _uvPixels; i++ )        
        {
            if (abs( pGrid[i] ) < min)
                min = abs( pGrid[i] );
            if (abs( pGrid[i] ) > max)
                max = abs( pGrid[i] );
        }
        
        // add 1% allowance to max - we don't want saturation.
        max = ((max - min) * 1.01) + min;
        
        // construct the image.
        image = (unsigned char *) malloc( _uvPixels * _uvPixels * sizeof( unsigned char ) );
        for ( int i = 0; i < _uvPixels * _uvPixels; i++ )
            image[i] = (unsigned char)( (abs( pGrid[i] ) - min) * ((double)256 / max) );
        
        // write the data.
        if (ok == 1)
        {
            
            size_t num_written = fwrite( image, 1, _uvPixels * _uvPixels, outputFile );
            if (num_written != (_uvPixels * _uvPixels)){
                ok = 0;
            }
        }
        
        // close file.
        fclose( outputFile );
        
    }
    
    // cleanup memory.
    free( (void *) fileHeader );
    if (image != NULL){
        free( image );
    }
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

int generateImageOfConvolutionFunction( char * pDeconvolutionFilename )
{

    int ok(1);

    // create the deconvolution image.
    cx_rowvec _deconvolutionImage(_uvPixels * _uvPixels);

    // create a single visibility with a value of 1.
    cx_rowvec tmpVisibility(1);
    tmpVisibility[0] = 1.0;

    // create a single uvw vector with a value of <0, 0, 0>.
    mat tmpSample(1, 3, fill::zeros);

    // generate the deconvolution function by gridding a single visibility.
    ok = gridVisibilities( _deconvolutionImage, tmpVisibility, 1, _aaKernelSize, _aaSupport, _aaKernel, 1, tmpSample, 1, 1 );

    // FFT the gridded data to get the deconvolution map.
    performFFT( _deconvolutionImage, _uvPixels, INVERSE );

    // save the deconvolution image.
    ok = saveBitmap( pDeconvolutionFilename, _deconvolutionImage );

    // return success flag.
    return ok;

} // generateImageOfConvolutionFunction
