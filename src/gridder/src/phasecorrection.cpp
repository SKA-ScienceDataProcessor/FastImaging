//
//	uvfacetting.c
//
//	Chris Skipper
//	24/07/2015
//
//	Changing the phase position on the sky from the initial phase position (or pointing position of the telescope) can be done
//	in two ways. The first (which we call UV facetting), applies a phase correction to each visibility in order that the phase centre
//	is shifted around the celestial sphere to a new position. This method is initiated by setting the 'uv_projection' parameter
//	in the 'uvfacetting-params' file to 'N'.
//
//	The second method (which we call image plane facetting) also uses a phase correction, but in addition will correct the
//	uvw coordinates in order that the resulting image is projected onto a new tangential plane centred on the new phase position. This
//	method is initiated by setting the 'uv_projection' parameter to 'Y'. We use the equations from Sault et al 1996 A&As 120 375-384 to
//	do this reprojection.
//
//                Image plane facetting                       UV facetting
//
//              ---------/---------------                   ----------------
//                     / oo IN oo                               oo IN oo
//                   / oo        ooo                    OUT  ooo        ooo
//                 / OUT            oo            ----------------         oo
//               / o                  o                   o                  o
//             /  o                    o                 o                    o
//                o                    o                 o                    o
//
//	The input and output UVW coordinate systems are aligned such that w points directly at the relevant phase position, and u points
//	directly east when the phase centre is at the local meridian, and the v-axis completes the cross product.
//
//	This program rotates the coordinate systems, and the supplied uvw vector, such that either the input or output coordinate systems
//	or the coordinate system of the celestial sphere can be aligned with the chosen xyz cartesian axes.
//
//	The UVW coordinate systems can be aligned with xyz such that u is along the x-axis, v is along the y-axis, and w is along the z-axis.
//	When the coordinate system of the celestial sphere is aligned with xyz then the north celestial pole (NCP) points directly along
//	the positive z-axis, the point defined by (RA 0, dec 0) points directly along the positive x-axis, and the point defined by
//	(RA 90, dec 0) points directly along the y-axis.
//
//	All rotation matrices used in this code are left-hand. i.e. for an initial uvw vector x, a rotated uvw vector x' and a rotation
//	matrix M, we use x' = Mx. All rotations are anti-clockwise when viewed from a direction such that the normal 2d cartesian axes X
//	and Y are formed by XY, XZ or YZ respectively. To rotate from a UVW coordinate system to world coordinates we first do a latitude
//	rotation about the x-axis by (90 - latitude) so that the phase position moves from (0,0,1) to some point at RA -90. We then do a
//	longitude rotation about the z-axis by (90 + longitude) so that the phase position moves to the required celestial coordinates.
//

#include "phasecorrection.h"

//
//	GENERAL FUNCTIONS
//


/*! toUppercase
 * \brief function to convert a string to uppercase
 * 
 * \param[in]  pChar string to be converted
 */

void PhaseCorrection::toUppercase( char * pChar )
{
    for ( char * ptr = pChar; *ptr; ptr++ )
        *ptr = toupper( *ptr );
	
} // toUppercase



//
//	MATRIX FUNCTIONS
//


/*! rotateX
 * \brief Construct a 3x3 matrix to rotate a vector about the X-axis.
 * 
 * \param[in]  pAngle angle of rotation
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::rotateX( double pAngle )
{
	
    pAngle = DEG_TO_RAD( pAngle );
    mat33 rotationMatrix = { 1,          0    ,               0,
                             0, cos(  pAngle ), -sin(  pAngle ),
                             0, sin(  pAngle  ), cos(  pAngle ),
                           };
	
    return rotationMatrix;
	
} 


/*! rotateY
 * \brief Construct a 3x3 matrix to rotate a vector about the Y-axis.
 * 
 * \param[in]  pAngle angle of rotation
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::rotateY( double pAngle )
{
    pAngle = DEG_TO_RAD( pAngle );
	
    mat33 rotationMatrix = { cos( pAngle ), 0, -sin(  pAngle ),
                             0            , 1,          0     ,
                             sin( pAngle ), 0,  cos( pAngle ) ,
                           };
	
	return rotationMatrix;
	
} 


/*! rotateZ
 * \brief Construct a 3x3 matrix to rotate a vector about the Z-axis.
 * 
 * \param[in]  pAngle angle of rotation
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::rotateZ( double pAngle )
{
    pAngle= DEG_TO_RAD( pAngle );
	
    mat33 rotationMatrix = { cos( pAngle ), -sin( pAngle ), 0,
                             sin( pAngle) , cos( pAngle ) , 0,
                               0          , 0             , 1,
                           };

    return rotationMatrix;
	
} 


/*! convertXYZtoUVW
 * \brief Constructs a rotation matrix that converts coordinates from world coordinates (X = RA 0, dec 0, Y = RA 90, dec 0, Z = RA 0,
 *        dec 90) into UVW coordinates (X = U, Y = V, Z = W).
 * 
 * \param[in]  pCoords polar coordinates
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::convertXYZtoUVW( PolarCoords pCoords )
{

    mat33 rotationMatrix = rotateZ( -(90 + pCoords.longitude) );

    return rotateX( -(90 - pCoords.latitude) ) * rotationMatrix;
	
} // convertXYZtoUVW


/*! convertUVWtoXYZ
 * \brief Constructs a rotation matrix that converts coordinates from UVW coordinates (X = U, Y = V, Z = W) into world coordinates
 *        (X = RA 0, dec 0, Y = RA 90, dec 0, Z = RA 0, dec 90).
 * 
 * \param[in]  pCoords polar coordinates
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::convertUVWtoXYZ( PolarCoords pCoords )
{

    mat33 rotationMatrix = rotateX( 90 - pCoords.latitude );
	
    return  rotateZ( pCoords.longitude + 90 ) * rotationMatrix;
	
} 




//
//	EPOCH CONVERSION FUNCTIONS
//


/*! epochConversionMatrix
 * \brief Constructs a rotation matrix that converts coordinates from one epoch to another. This is done using a longitude rotation, a latitude rotation, and
 *        another longitude rotation. The three rotation angles are specified as constants at the top of this program, and can be easily found using an online tool
 *        such as NED (https://ned.ipac.caltech.edu/forms/calculator.html). Using NED, simply convert a position at RA 0, DEC 90 from one epoch to another and the
 *        rotation angles are given as the output coordinates (RA, DEC, PA).
 * 
 * \param[in] pNP_RA
 * \param[in] pNP_DEC
 * \param[in] pNP_RA_OFFSET
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET )
{

    mat33 rotationMatrix =  rotateY( 90 - pNP_DEC ) * rotateZ( -pNP_RA ) ;
    rotationMatrix =  rotateZ( pNP_RA_OFFSET ) * rotationMatrix;

    return rotationMatrix;
}



/*! doEpochConversion
 * \brief Construct a matrix that does epoch conversion between two positions. We simply compare the input and output
 *        epoch, and then construct a suitable rotation matrix.
 * 
 * \param[in] pIn
 * \param[in] pOut
 * \param[out] rotationMatrix 3x3 matrix 
 */

mat33 PhaseCorrection::doEpochConversion( PolarCoords pIn, PolarCoords pOut )
{

    // default to no epoch conversion.
    mat33 epochConversion = { 1, 0, 0,
                              0, 1, 0,
                              0, 0, 1,
                            };

    // J2000 to/from galactic.
    if ((strcmp( pIn.epoch, "J2000" ) == 0) && (strcmp( pOut.epoch, "GALACTIC" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_J2000, NP_DEC_GAL_IN_J2000, NP_RA_OFFSET_GAL_IN_J2000 );
    if ((strcmp( pIn.epoch, "GALACTIC" ) == 0) && (strcmp( pOut.epoch, "J2000" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_GAL, NP_DEC_J2000_IN_GAL, NP_RA_OFFSET_J2000_IN_GAL );
	
    // B1950 to/from galactic.
    if ((strcmp( pIn.epoch, "B1950" ) == 0) && (strcmp( pOut.epoch, "GALACTIC" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_B1950, NP_DEC_GAL_IN_B1950, NP_RA_OFFSET_GAL_IN_B1950 );
    if ((strcmp( pIn.epoch, "GALACTIC" ) == 0) && (strcmp( pOut.epoch, "B1950" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_GAL, NP_DEC_B1950_IN_GAL, NP_RA_OFFSET_B1950_IN_GAL );
	
    // B1950 to/from J2000.
    if ((strcmp( pIn.epoch, "B1950" ) == 0) && (strcmp( pOut.epoch, "J2000" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_B1950, NP_DEC_J2000_IN_B1950, NP_RA_OFFSET_J2000_IN_B1950 );
    if ((strcmp( pIn.epoch, "J2000" ) == 0) && (strcmp( pOut.epoch, "B1950" ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_J2000, NP_DEC_B1950_IN_J2000, NP_RA_OFFSET_B1950_IN_J2000 );
	
    return epochConversion;
	
} 


//
//	PHASE-CORRECTION FUNCTIONS
//

/*! getPathLengthDifference
 * \brief Rotate the uvw coordinates from the input to the output coordinate system, and determine the change in the path length
 *        in order that the visibility phase delay can be calculated.
 * 
 * \param[in] pUVW
 * \param[out] pathLengthDifference 
 */

double PhaseCorrection::getPathLengthDifference( vec3 pUVW )
{

    vec3 inUVW = {0, 0, 1};
    vec3 phaseI = _uvwRotation* inUVW;
	
    // now the path length difference is given by the dot product of our newly constructed vector with uvw.
    return dot((phaseI - inUVW), (_uvwRotation * pUVW));
	
} // getPathLengthDifference



/*! reprojectUV
 * \brief Do a full uv-reprojection on the UVW coordinates. This code reproduces the equations shown in Appendix A of
 *        Sault et al 1996 A&ASS 120 375-384. In the image domain we require a rotational transform R from the input to the output
 *        coordinate system, and in the UV-domain the required matrix is R^-1[T] (eqn A4).
 * 
 * \param[out] rotationMatrixReduced 
 */

mat22 PhaseCorrection::reprojectUV()
{

    // extract the top-left 2x2 matrix from _uvwRotation.
    mat22 rotationMatrixReduced= _uvwRotation(span(0, 0), span(1, 1));

    // get the 2x2 inverse and then the 2x2 transpose.
    rotationMatrixReduced = trans( inv( rotationMatrixReduced ) );
	
    return rotationMatrixReduced;
	
} // reprojectUV


/*! init
 * \brief Prepare the epoch conversion and rotation matrices.
 * 
 */

void PhaseCorrection::init()
{
    // prepare the epoch conversion matrix.
    _epochConversion = doEpochConversion( inCoords, outCoords );
    
    // calculate the uvw rotation matrix that transforms the uvw coordinates relative to the in phase centre to uvw
    // coordinates that are relative to the out phase centre. we use latitude and longitude rotations about
    // both the in and out phase positions, and include an epoch-conversion step in between.
    _uvwRotation = convertUVWtoXYZ( inCoords );
	
    // now do epoch conversion.
    _uvwRotation = _epochConversion * _uvwRotation;
	
    // finally, rotate into UVW coordinates relative to the output position.
    _uvwRotation =  convertXYZtoUVW( outCoords ) * _uvwRotation;
  
} 


/*! rotate
 * \brief Get the path length difference and rotated uvw coordinates.
 * 
 */

void PhaseCorrection::rotate()
{

    vec coluvwIn = conv_to< vec >::from(uvwIn);
    vec coluvwOut;

    // get the additional path length difference when switching from the in to the out phase centre..
    phase = getPathLengthDifference( coluvwIn );
    
    // are we are doing full uv reprojection? If not, then just convert to output coordinates.
    coluvwOut = _uvwRotation * coluvwIn;

    //  cout << "rotation" << _uvwRotation << endl;
    if (uvProjection == 1) {
        coluvwOut = reprojectUV() * coluvwOut ;
    }

    uvwOut = conv_to< rowvec >::from(coluvwOut);

} 
