#!/bin/bash

# Generate fftw wisdom files for power-of-two image sizes

# Do not use system executables of FFTW-Wisdom
# Use instead the STP compiled FFTW-Wisdom executable

MAX_SIZE="16384"
TYPE_1D_FFT="kof"
TYPE_R2C_2D_FFT="rob"
TYPE_CX_2D_FFT="cof"
BUILDTYPE="Debug"
EXEC_PATH="third-party/fftw/bin/"
EXEC="fftw-wisdom"
OUTDIR="wisdomfiles"
OUTFILE="WisdomFile_STP.fftw"
NPROC=`nproc`

while getopts "e:s:drifph" OPTION
do
	case $OPTION in
		e)
			EXEC_PATH="$OPTARG"
			;;
		s)
			MAX_SIZE="$OPTARG"
			;;
		d)
			BUILDTYPE="Debug"
			# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
			EXEC_PATH="../../build/$BUILDTYPE/third-party/fftw/bin/"
			;;
		r)
			BUILDTYPE="Release"
			# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
			EXEC_PATH="../../build/$BUILDTYPE/third-party/fftw/bin/"
			;;
		i)
			BUILDTYPE="RelWithDebInfo"
			# This is the default build folder created by the build.sh script. Change it if you built STP into a different directory.
			EXEC_PATH="../../build/$BUILDTYPE/third-party/fftw/bin/"
			;;
		f)
			EXEC="fftwf-wisdom"
			;;
		p)
			TYPE_1D_FFT="kif"
			TYPE_R2C_2D_FFT="rib"
			TYPE_CX_2D_FFT="cif"
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
			echo " -f    Use fftwf-wisdom (single-precision) executable (default is double-precision fftw-wisdom)"
			echo " -p    Use in-place FFT rather than out-of-place FFT (default is out-of-place FFT)"
			echo " -s <max_size> Specify maximum image size. E.g.: -s \"4096\""
			echo 
			echo "Notes:"
			echo " To indicate the location of the executable use only one of the following: -e, -d, -r, -i"
			echo " If FFTW is compiled in Float mode add the -f option."
			echo " Possible values for -s option:"
			echo " 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"
			echo
			exit 1
			;;
	esac
done


# Set full path
EXECUTABLE=$EXEC_PATH/$EXEC

if [ ! -f $EXECUTABLE ] ; then
	echo "ERROR! Executable not found: $EXECUTABLE"
	exit 1
fi

if [ -d ${OUTDIR} ] ; then
	rm -r ${OUTDIR}
fi
mkdir $OUTDIR

SIZES="2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"
FFTWSTR1=
FFTWSTR2=
FFTWSTR3=

for IMGSIZE in $SIZES ; do
	if [ $IMGSIZE -le $MAX_SIZE ] ; then
		FFTWSTR1="${FFTWSTR1} ${TYPE_1D_FFT}${IMGSIZE}"
		FFTWSTR2="${FFTWSTR2} ${TYPE_R2C_2D_FFT}${IMGSIZE}x${IMGSIZE}"
		FFTWSTR3="${FFTWSTR3} ${TYPE_CX_2D_FFT}${IMGSIZE}x${IMGSIZE}"
	fi
done

echo "${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE} ${FFTWSTR1} ${FFTWSTR2} ${FFTWSTR3}"
${EXECUTABLE} -n -m -T ${NPROC} -o ${OUTDIR}/${OUTFILE} ${FFTWSTR1} ${FFTWSTR2} ${FFTWSTR3}

exit 0
