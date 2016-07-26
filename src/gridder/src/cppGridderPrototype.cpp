#include "gridderprototype.h"
//
//	main()
//
//	CJS: 07/08/2015
//
//	Main processing.
//



int main(int pArgc, char ** pArgv )
{


    char DIRTY_BEAM_EXTENSION[] = "-dirty-beam.bmp";
    char CLEAN_BEAM_EXTENSION[] = "-clean-beam.bmp";
    char GRIDDED_EXTENSION[] = "-gridded.bmp";
    char DIRTY_IMAGE_EXTENSION[] = "-dirty-image.bmp";
    char CLEAN_IMAGE_EXTENSION[] = "-clean-image.bmp";
    char RESIDUAL_IMAGE_EXTENSION[] = "-residual-image.bmp";
    char DECONVOLUTION_EXTENSION[] = "-deconvolution.bmp";
    char UVW_PREFIX[] = "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/input-data/uvw-";
    char DATA_PREFIX[] = "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/input-data/data-";
    char CHANNEL_PREFIX[] = "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/input-data/channel-";
    char INPUT_EXTENSION[] = ".txt";


    char outputGriddedFilename[ 100 ];
    char outputDirtyImageFilename[ 100 ];
    char outputDeconvolutionFilename[ 100 ];

    char inputUVWFilename[100];
    char inputVisFilename[100];
    char inputChannelFilename[100];
    char * inputIdentifier = pArgv[1];
    char * filenamePrefix = pArgv[2];


    // read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
    // read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
    if (pArgc != 3)
    {
        printf("Wrong number of arguments. Require the input input identifier and bitmap prefix.\n");
        return 1;
    }

   /*char inputUVWFilename[1000]  = "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/input-data/uvw-vla-sim.txt";
    char inputVisFilename[1000] = "/home/ca-santos/SKADEV/svn/implementation/experiments/gridding/input-data/data-vla-sim.txt";
   char inputChannelFilename[1000]= "ola"; */

    // build filenames.
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


    // create the grid.
    cx_rowvec _grid;
    _grid.set_size(_uvPixels* _uvPixels);
    _grid.fill(0);

    wall_clock timer;
    double n =0.0;




    // load data. we need to load 'samples' (the uvw coordinates), 'channels' (frequency of each channel) and 'visibilities' (one for each sample/channel combination).
    int ok = loadData( inputUVWFilename, inputVisFilename, inputChannelFilename );


    if (ok == 1)
    {

     timer.tic();
       cout << "\n--- time (load data): " << timer.toc() << endl;

        // calculate kernel size and other parameters.
        calculateKernelSize();

       cout << "\n--- time (calculate kernel size):  " << timer.toc() << endl;

        timer.tic();
        // generate kernel.
        ok = generateKernel();

        cout <<  "--- time (generate kernel): " << timer.toc() << endl;

    }


    timer.tic();
    // do phase correction.
    doPhaseCorrection();

    cout << "--- time (phase rotation):  " << timer.toc() << endl;


   if (ok == 1)
    {

        timer.tic();
        // generate image of convolution function.
        ok = generateImageOfConvolutionFunction( outputDeconvolutionFilename );

        cout << "--- time (generate deconvolution map):  " << timer.toc() << endl;

    }

    if (ok == 1)
    {

        // grid visibilities.
        printf( "\ngridding visibilities for dirty image.....\n" );
        timer.tic();
        ok = gridVisibilities( _grid, _visibility, _oversample, _kernelSize, _support, _kernel, _wPlanes, _sample, _numSamples, _numChannels );


        cout << "--- time (gridding for dirty image):  " << timer.toc() << endl;

    }

    if (ok == 1)
    {

        // save image.
        printf( "pixel intensity range of uv data:\n" );

        timer.tic();
        ok = saveBitmap( outputGriddedFilename, _grid );

        cout <<  "\n--- time (save gridded image):  " << timer.toc() << endl;

        timer.tic();


        // make dirty image.
        performFFT( _grid, _uvPixels, INVERSE );



        cout << "\n--- time (fft):  " << timer.toc() << endl;

        // normalise the image following the FFT.
            _grid /= round(_uvPixels * _uvPixels);

         //   cout << "_grid " << _grid << endl;

          //  cout << "end _grid" << endl;


        // divide dirty image by deconvolution image.
        double maxValue; //= max(abs(_grid));
        double deconvAbs = 0.0;
        cx_double gridVal = 0.0;

        for ( int i = 0; i < _uvPixels; i++ )
            for ( int j = 0; j < _uvPixels; j++ )
            {
                deconvAbs = abs( _deconvolutionImage[ (j * _uvPixels) + i ] );
                gridVal = _grid[ (j * _uvPixels) + i ];
                if (deconvAbs == 0) {
                    gridVal = 0;
                }
                else {
                    gridVal /= deconvAbs;
                 }
            }

        maxValue = max(abs(_grid));

        printf( "deconvolved image max value: %f\n", maxValue );

        // save dirty image.
        printf( "\npixel intensity range of dirty image:\n" );
        timer.tic();
        ok = saveBitmap( outputDirtyImageFilename, _grid );

        cout <<  "--- time (save dirty image):  " << timer.toc() << endl;

    }

    return true;

} // main
