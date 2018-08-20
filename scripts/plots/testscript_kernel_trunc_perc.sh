#!/bin/bash

if [ $# = 0 ]
then
	echo "Missing title parameter."
	exit 1
fi

FIELD="kernel_trunc_perc"
CONFFILE="fastimg_awproj_config.json"
RESFILE="res_"$1"_"$FIELD".txt"


echo_run()
{
	echo "$1"
	eval "$1"
}

echo > $RESFILE

for i in 1.0 0.1 0.01 0.001 0.0 
do
	echo
	echo "$FIELD" = $i
	sed -i "s|\"$FIELD.*|\"$FIELD\": $i,|g" $CONFFILE
	echo_run "./reduce fastimg_awproj_config.json simdata_nstep10.npz out.json -l | grep ' = ' | awk 'BEGIN {printf $i}{printf \" %s\", \$NF} END {printf \"\\n\"}' > tmp_$RESFILE"
	cat $RESFILE tmp_$RESFILE > out_$RESFILE
	mv out_$RESFILE $RESFILE
	rm tmp_$RESFILE
done

exit 0
