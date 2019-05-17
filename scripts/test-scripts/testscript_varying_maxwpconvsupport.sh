#!/bin/bash

if [ $# != 2 ]
then
	echo "Missing config file and npz measurement set"
	exit 1
fi

FIELD="max_wpconv_support"
PARLIST="63 127 255 511 1023"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1 $2

exit 0
