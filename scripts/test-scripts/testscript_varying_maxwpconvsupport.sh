#!/bin/bash

if [ $# = 0 ]
then
	echo "Missing config label."
	exit 1
fi

FIELD="max_wpconv_support"
PARLIST="15 30 60 120 250 500 1000"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1

exit 0
