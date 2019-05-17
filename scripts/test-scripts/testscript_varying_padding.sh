#!/bin/bash

if [ $# != 2 ]
then
	echo "Missing config file and npz measurement set"
	exit 1
fi

FIELD="padding_factor"
PARLIST="1.0 1.2 1.4 1.6 1.8 2.0"

script_full_path=$(dirname "$0")
$script_full_path/base_testscript.sh $FIELD "$PARLIST" $1 $2

exit 0
