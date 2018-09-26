#!/bin/bash

if [ $# = 0 ]
then
	echo "Missing config label."
	exit 1
fi

FIELD="undersampling_opt"
PARLIST="1 2 3 4"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1

exit 0
