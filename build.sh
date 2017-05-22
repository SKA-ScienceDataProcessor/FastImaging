#!/bin/bash

NUMCORES=$(nproc)

# Default build type is Debug
BUILDTYPE="Debug"
# Default USE_FLOAT=OFF
USEFLOAT="OFF"

while getopts "drifh" OPTION
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
		h)
			echo 
			echo "Build STP prototype, generate FFTW plans and run tests."
			echo "Usage: ./build.sh <OPTIONS>"
			echo
			echo "Available OPTIONS:"
			echo " -d    Use BUILDTYPE=Debug (default)"
			echo " -r    Use BUILDTYPE=Release"
			echo " -i    Use BUILDTYPE=RelWithDebInfo"
			echo " -f    Use USE_FLOAT=ON (default is USE_FLOAT=OFF)"
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
echo 
echo " >> Create build directory in build/$BUILDTYPE"
if [ -d build/$BUILDTYPE ] ; then
	rm -r build/$BUILDTYPE
fi

mkdir -p build/$BUILDTYPE
cd build/$BUILDTYPE

# Run CMake
COMMAND="cmake -DCMAKE_BUILD_TYPE="$BUILDTYPE" -DUSE_FLOAT=${USEFLOAT} ../../src/"
echo
echo " >> Run cmake and compile"
echo $COMMAND
$COMMAND

# Build
make all -j $NUMCORES

# Generate FFTW plans
echo
echo " >> Generate FFTW plans using fftw-wisdom method"
cd ../../scripts/fftw-wisdom

if [[ ${USEFLOAT} == "ON" ]]
then
	./generate_wisdom_f.sh
else
	./generate_wisdom.sh
fi

echo
echo " >> Run tests"

# Run tests
cd ../../build/$BUILDTYPE
make test

