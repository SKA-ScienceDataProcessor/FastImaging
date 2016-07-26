#ifndef _PHASECORRECTION_H
#define _PHASECORRECTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <fftw3.h>
#include <iostream>
#include <armadillo>
#include <locale>         // std::locale, std::toupper



#define DEG_TO_RAD(x) (x * (M_PI/180.0))
#define RAD_TO_DEG(x) (x * 180.0/M_PI)

using namespace std;
using namespace arma;


//
//	STRUCTURES
//

// latitude and logitude coordinates.
struct PolarCoords
{
	double longitude;
	double latitude;
	char epoch[20];
};
typedef struct PolarCoords PolarCoords;


class PhaseCorrection
{
	
	public:

//
//	GLOBAL VARIABLES
//
	
		// input and output uvw vectors.
        rowvec3 uvwIn;
        rowvec3 uvwOut;
		
		double phase;
			
		// input and output phase centres.
		PolarCoords inCoords;
		PolarCoords outCoords;
		
		// are we doing full uv reprojection, or just calculating the phase change?
        int uvProjection;
		
//
//	PUBLIC FUNCTIONS
//
		
		void init();
		void rotate();
		
	private:

//
//	CONSTANTS
//
	
		// coordinates of the Galactic coordinate system north pole in the J2000 coordinate system.
        static constexpr double NP_RA_GAL_IN_J2000 = 192.859496;
        static constexpr double NP_DEC_GAL_IN_J2000 = 27.128353;
        static constexpr double NP_RA_OFFSET_GAL_IN_J2000 = 302.932069;
	
		// coordinates of the J2000 coordinate system north pole in the galactic coordinate system.
        static constexpr double NP_RA_J2000_IN_GAL = 122.932000;
        static constexpr double NP_DEC_J2000_IN_GAL = 27.128431;
        static constexpr double NP_RA_OFFSET_J2000_IN_GAL = 12.860114;
	
		// coordinates of the Galactic coordinate system north pole in the B1950 coordinate system.
        static constexpr double NP_RA_GAL_IN_B1950 = 192.250000;
        static constexpr double NP_DEC_GAL_IN_B1950 = 27.400000;
        static constexpr double NP_RA_OFFSET_GAL_IN_B1950 = 303.000000;
	
		// coordinates of the B1950 coordinate system north pole in the galactic coordinate system.
        static constexpr double NP_RA_B1950_IN_GAL = 123.000000;
        static constexpr double NP_DEC_B1950_IN_GAL = 27.400000;
        static constexpr double NP_RA_OFFSET_B1950_IN_GAL = 12.250000;
	
		// coordinates of the J2000 coordinate system north pole in the B1950 coordinate system.
        static constexpr double NP_RA_J2000_IN_B1950 = 359.686210;
        static constexpr double NP_DEC_J2000_IN_B1950 = 89.721785;
        static constexpr double NP_RA_OFFSET_J2000_IN_B1950 = 0.327475;
	
		// coordinates of the B1950 coordinate system north pole in the J2000 coordinate system.
        static constexpr double NP_RA_B1950_IN_J2000 = 180.315843;
        static constexpr double NP_DEC_B1950_IN_J2000 = 89.72174782;
        static constexpr double NP_RA_OFFSET_B1950_IN_J2000 = 179.697628;

//
//	GLOBAL VARIABLES
//

		// the full rotation matrix from the input to the output coordinate system. this matrix would need
		// to be calculated once and then stored so that it can be applied to many baseline vectors.
        mat33 _uvwRotation;

		// rotation matrix for epoch conversion.
        mat33 _epochConversion;

//
//	PRIVATE FUNCTIONS
//
		
        void toUppercase( char * pChar );
        mat33 rotateX( double pAngle );
        mat33 rotateY( double pAngle );
        mat33 rotateZ( double pAngle );
        mat33 convertXYZtoUVW( PolarCoords pCoords );
        mat33 convertUVWtoXYZ( PolarCoords pCoords );
        mat33 epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET );
        mat33 doEpochConversion( PolarCoords pIn, PolarCoords pOut );
        double getPathLengthDifference( vec3 pUVW );
        mat22 reprojectUV();
	
}; // PhaseCorrection

#endif // _PHASECORRECTION_H
