#!/bin/bash

NUMCORES=$(nproc)

# Default build type is Debug
BUILDTYPE="Debug"
# Default USE_FLOAT=OFF
USEFLOAT="OFF"
# Disable FFT shift
FFTSHIFT="OFF"
# Generate fftw plans
GENFFT=1

while getopts "drifsnh" OPTION
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
			echo " -n    Do not generate fftw wisdom files"
			echo
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
echo 
echo " >> Create build directory in build/$BUILDTYPE"
if [ -d build/$BUILDTYPE ] ; then
	rm -r build/$BUILDTYPE
fi

mkdir -p build/$BUILDTYPE
cd build/$BUILDTYPE

# Run CMake
COMMAND="cmake -DCMAKE_BUILD_TYPE="$BUILDTYPE" -DUSE_FLOAT=${USEFLOAT} -DUSE_FFTSHIFT=${FFTSHIFT} ../../src/"
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

