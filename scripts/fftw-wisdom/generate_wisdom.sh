#!/bin/bash

# Generate fftw wisdom files for various image sizes

# Do not use system executables of FFTW-Wisdom
# Use instead the STP compiled FFTW-Wisdom executable

BUILDTYPE="Debug"
EXEC="fftw-wisdom"
OUTDIR="wisdomfiles"

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
			EXEC="fftwf-wisdom"
			OUTDIR="wisdomfiles_f"
			;;
		h)
			echo 
			echo "Generate fftw wisdom files for various image sizes."
			echo "Usage: ./generate_wisdom.sh <OPTIONS>"
			echo
			echo "Available OPTIONS:"
			echo " -d    Use BUILDTYPE=Debug (default)"
			echo " -r    Use BUILDTYPE=Release"
			echo " -i    Use BUILDTYPE=RelWithDebInfo"
			echo " -f    Use fftwf-wisdom (single-precision) executable (default is fftw-wisdom)"
			echo
			exit 1
			;;
	esac
done

# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
EXECUTABLE=../../build/$BUILDTYPE/third-party/fftw/bin/$EXEC

if [ ! -f $EXECUTABLE ] ; then
	echo "ERROR! Executable not found: $EXECUTABLE"
	exit 1
fi

if [ -d ${OUTDIR} ] ; then
	rm -r ${OUTDIR}
fi
mkdir $OUTDIR

for IMGSIZE in 128 256 512 1024 2048 4096 8192 16384 32768 65536 ; do

NPROC=`nproc`
FFTWSTR1="rob${IMGSIZE}x${IMGSIZE}"
OUTFILE1="WisdomFile_${FFTWSTR1}.fftw"
FFTWSTR2="rib${IMGSIZE}x${IMGSIZE}"
OUTFILE2="WisdomFile_${FFTWSTR2}.fftw"

echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}
echo
echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}
echo
done
