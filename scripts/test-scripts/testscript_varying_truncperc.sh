#!/bin/bash

if [ $# != 2 ]
then
	echo "Missing config file and npz measurement set"
	exit 1
fi

FIELD="kernel_trunc_perc"
PARLIST="20.0 10.0 5.0 1.0 0.1 0.01 0"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1 $2

exit 0
