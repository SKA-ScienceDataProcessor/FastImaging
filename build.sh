#!/bin/bash


# Default build type is Debug
BUILDTYPE="Debug"

if [[ $1 == "r" ]] ; then
   BUILDTYPE="Release"
fi

# Cleanup any preexisting build files
if [ -d build/$BUILDTYPE ] ; then
	rm -r build/$BUILDTYPE
fi

mkdir -p build/$BUILDTYPE
cd build/$BUILDTYPE

# Run CMake
cmake -DCMAKE_BUILD_TYPE="$BUILDTYPE" ../../src/

# Build
make all -j8

# Run tests
make test

