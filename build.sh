#!/bin/bash

NUMCORES=$(nproc)

# Default build type is Debug
BUILDTYPE="Debug"
# Default USE_FLOAT=OFF
USEFLOAT="OFF"
# Disable FFT shift
FFTSHIFT="OFF"
# Use Serial Gridder
SERIALGRIDDER="OFF"
# Generate fftw plans
GENFFT=1
# Enable W-projection
ENABLE_WPROJECTION="OFF"
# Enable A-projection
ENABLE_APROJECTION="OFF"
# Enable STP debug
ENABLE_STP_DEBUG="OFF"

while getopts "drifsgnwlh" OPTION
do
	case $OPTION in
		d)
			BUILDTYPE="Debug"
			;;
		r)
			BUILDTYPE="Release"
			;;
		i)
			BUILDTYPE="RelWithDebInfo"
			;;
		f)
			USEFLOAT="ON"
			;;
		s)
			FFTSHIFT="ON"
			;;
		g)
			SERIALGRIDDER="ON"
			;;
		w)
			ENABLE_WPROJECTION="ON"
			;;
		a)
			ENABLE_APROJECTION="ON"
			;;
		l)
			ENABLE_STP_DEBUG="ON"
			;;
		n)
			GENFFT=0
			;;
		h)
			echo 
			echo "Build STP prototype, generate FFTW wisdom files and run tests."
			echo "Usage: ./build.sh <OPTIONS>"
			echo
			echo "Available OPTIONS:"
			echo " -d    Set BUILDTYPE=Debug (default)"
			echo " -r    Set BUILDTYPE=Release"
			echo " -i    Set BUILDTYPE=RelWithDebInfo"
			echo " -f    Set USE_FLOAT=ON (default is USE_FLOAT=OFF)"
			echo " -s    Set USE_FFTSHIFT=ON (default is USE_FFTSHIFT=OFF)"
			echo " -g    Set USE_SERIAL_GRIDDER=ON (default is USE_SERIAL_GRIDDER=OFF)"
			echo " -w    Enable support for W-projection (default is ENABLE_WPROJECTION="OFF")"
			echo " -a    Enable support for A-projection (default is ENABLE_APROJECTION="OFF")"
			echo " -l    Enable debug logger on STP library (default is ENABLE_STP_DEBUG="OFF")"
			echo " -n    Do not generate fftw wisdom files"
			echo
			exit 1
			;;
		\?)
			exit 1
      			;;
	esac
done


# Cleanup any preexisting build files
echo
echo " ***************** BUILD STP *****************"
echo " >> BUILDTYPE = ${BUILDTYPE}"
echo " >> USE_FLOAT = ${USEFLOAT}"
echo " >> USE_FFTSHIFT = ${FFTSHIFT}"
echo " >> USE_SERIAL_GRIDDER = ${SERIALGRIDDER}"
echo " >> ENABLE_WPROJECTION = ${ENABLE_WPROJECTION}"
echo " >> ENABLE_APROJECTION = ${ENABLE_APROJECTION}"
echo " >> ENABLE_STP_DEBUG = ${ENABLE_STP_DEBUG}"
echo 
echo " >> Create build directory in build/$BUILDTYPE"
if [ -d build/$BUILDTYPE ] ; then
	rm -r build/$BUILDTYPE
fi

mkdir -p build/$BUILDTYPE
cd build/$BUILDTYPE

# Run CMake
COMMAND="cmake -DCMAKE_BUILD_TYPE="$BUILDTYPE" -DUSE_FLOAT=${USEFLOAT} -DUSE_FFTSHIFT=${FFTSHIFT} -DUSE_SERIAL_GRIDDER=${SERIALGRIDDER} \
         -DENABLE_WPROJECTION=${ENABLE_WPROJECTION} -DENABLE_APROJECTION=${ENABLE_APROJECTION} -DENABLE_STP_DEBUG=${ENABLE_STP_DEBUG} \
	 ../../src/"
echo
echo " >> Run cmake and compile"
echo $COMMAND
$COMMAND

# Build
make all -j $NUMCORES

# Generate FFTW plans
if [ $GENFFT == 1 ] ; then
	echo
	echo " >> Generate FFTW wisdom files using fftw-wisdom tool"
	make fftwisdom
fi

echo
echo " >> Run tests"

# Run tests
cd ../../build/$BUILDTYPE
make test

exit $?
