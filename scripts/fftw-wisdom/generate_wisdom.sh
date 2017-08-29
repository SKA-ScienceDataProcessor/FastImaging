#!/bin/bash

# Generate fftw wisdom files for various image sizes

# Do not use system executables of FFTW-Wisdom
# Use instead the STP compiled FFTW-Wisdom executable

SIZES="128 256 512 1024 2048 4096 8192 16384 32768 65536"
BUILDTYPE="Debug"
EXEC="fftw-wisdom"
OUTDIR="wisdomfiles"
EXEC_PATH=

while getopts "e:s:drifh" OPTION
do
	case $OPTION in
		e)
			EXEC_PATH="$OPTARG"
			;;
        s)
			SIZES="$OPTARG"
			;;
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
			;;
		h)
			echo 
			echo "Generate fftw wisdom files for various image sizes."
			echo "Usage: ./generate_wisdom.sh <OPTIONS>"
			echo
			echo "Available OPTIONS:"
			echo " -e <exec_path> Path of the wisdom executable"
			echo " -d    Use Debug wisdom executable (default)"
			echo " -r    Use Release wisdom executable"
			echo " -i    Use RelWithDebInfo wisdom executable"
			echo " -f    Use fftwf-wisdom (single-precision) executable (default is fftw-wisdom)"
			echo " -s <sizes> Specify alternative image sizes (separated by white spaces). E.g.: -s \"1024 2048 4096\""
			echo 
			echo "Notes:"
			echo " To indicate the location of the executable use only one of the following: -e, -d, -r, -i"
			echo " If FFTW is compiled in Float mode add the -f option."
			echo " Use -s only if the needed image sizes are not in this list:"
			echo " 128 256 512 1024 2048 4096 8192 16384 32768 65536"
			echo
			exit 1
			;;
	esac
done

# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
EXECUTABLE=../../build/$BUILDTYPE/third-party/fftw/bin/$EXEC

if [ ! -z $EXEC_PATH ] ; then
	EXECUTABLE=$EXEC_PATH/$EXEC
fi

if [ ! -f $EXECUTABLE ] ; then
	echo "ERROR! Executable not found: $EXECUTABLE"
	exit 1
fi

if [ -d ${OUTDIR} ] ; then
	rm -r ${OUTDIR}
fi
mkdir $OUTDIR

for IMGSIZE in $SIZES ; do

NPROC=`nproc`
FFTWSTR1="rob${IMGSIZE}x${IMGSIZE}"
OUTFILE1="WisdomFile_${FFTWSTR1}.fftw"
echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE1} ${FFTWSTR1}

#FFTWSTR2="rib${IMGSIZE}x${IMGSIZE}"
#OUTFILE2="WisdomFile_${FFTWSTR2}.fftw"
#echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}"
#${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE2} ${FFTWSTR2}

done

exit 0
