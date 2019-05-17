#!/bin/bash

if [ $# != 2 ]
then
	echo "Missing config file and npz measurement set"
	exit 1
fi

FIELD="image_size_pix"
PARLIST="2048 4096 8192 16384"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1 $2

exit 0
