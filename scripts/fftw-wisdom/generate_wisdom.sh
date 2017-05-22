#!/bin/bash

# Generate wisdom for various image sizes

# Do not use system executables of FFTW-Wisdom
# Use instead the STP compiled FFTW-Wisdom executable
# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
EXECUTABLE=../../build/Release/third-party/fftw/bin/fftw-wisdom
OUTDIR=wisdomfiles

if [ -d ${OUTDIR} ] ; then
   rm -r ${OUTDIR}
fi
mkdir $OUTDIR

for IMGSIZE in 128 256 512 1024 1448 2048 2896 4096 5792 8192 11585 16384 23170 32768 65536 ; do

NPROC=`nproc`
FFTWSTR1="cof${IMGSIZE}x${IMGSIZE}"
OUTFILE1="WisdomFile_${FFTWSTR1}.fftw"
FFTWSTR2="rof${IMGSIZE}x${IMGSIZE}"
OUTFILE2="WisdomFile_${FFTWSTR2}.fftw"

echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}
echo
echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}
echo
done
