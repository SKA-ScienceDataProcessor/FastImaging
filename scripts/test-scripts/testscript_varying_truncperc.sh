#!/bin/bash

if [ $# = 0 ]
then
	echo "Missing config label."
	exit 1
fi

FIELD="kernel_trunc_perc"
PARLIST="10.0 1.0 0.1 0.01 0.001"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1

exit 0
