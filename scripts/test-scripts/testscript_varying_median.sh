#!/bin/bash

if [ $# = 0 ]
then
	echo "Missing config label."
	exit 1
fi

FIELD="wplanes_median"
PARLIST="false true"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1

exit 0
