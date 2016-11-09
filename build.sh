#!/bin/bash

# Build type is Debug 
mkdir -p build/debug
cd build/debug

# Cleanup any preexisting build files
rm -r *

# Run CMake
cmake ../../src/

# Build
make all

# Run tests
make test

