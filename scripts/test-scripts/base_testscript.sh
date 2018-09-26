#!/bin/bash

if [ $# != 3 ]
then
	echo "Missing Field, Parameter-list and Config-label."
	exit 1
fi

FIELD=$1
CONFFILE_ORIG="fastimg_wproj_wenssconfig.json"
RESFILE_BENCH="res_bench_"$3"_"$FIELD".txt"
RESFILE1="res_accu_s1_"$3"_"$FIELD".txt"
RESFILE2="res_accu_s2_"$3"_"$FIELD".txt"

echo_run()
{
	echo "$1"
	eval "$1"
}


echo > $RESFILE_BENCH
echo > $RESFILE1
echo > $RESFILE2
CONFFILE=tmp_$CONFFILE_ORIG
cp $CONFFILE_ORIG $CONFFILE

sed -i "s|\"sourcefind_detection.*|\"sourcefind_detection\": 30,|g" $CONFFILE
sed -i "s|\"sourcefind_analysis.*|\"sourcefind_analysis\": 30,|g" $CONFFILE

for i in $2
do
	echo
	echo "$FIELD" = $i
	if [[ $(cat $CONFFILE | grep $FIELD | grep ",") ]] ;
	then
		sed -i "s|\"$FIELD.*|\"$FIELD\": $i,|g" $CONFFILE
	else
		sed -i "s|\"$FIELD.*|\"$FIELD\": $i|g" $CONFFILE
	fi
	rm logfile.txt
	i=$(echo "${i//\"}")
	echo_run "./reduce $CONFFILE wenssdata_awproj.npz -l > /dev/null"
	echo_run "cat logfile.txt | grep ' \[benchmark\] ' | grep ' = ' | awk 'BEGIN {printf \"$i\"}{printf \" %s\", \$NF} END {printf \"\\n\"}' > tmp_$RESFILE_BENCH"
	echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'extremum_x_idx=8192, extremum_y_idy=8192' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {printf \"$i\"}{printf \" %s\", \$NF} END {printf \"\\n\"}' > tmp_$RESFILE1"
	echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'extremum_x_idx=1259[0-9], extremum_y_idy=340[0-9]' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {printf \"$i\"}{printf \" %s\", \$NF} END {printf \"\\n\"}' > tmp_$RESFILE2"
	
    cat $RESFILE_BENCH tmp_$RESFILE_BENCH > out_$RESFILE_BENCH
	mv out_$RESFILE_BENCH $RESFILE_BENCH
	rm tmp_$RESFILE_BENCH
	cat $RESFILE1 tmp_$RESFILE1 > out_$RESFILE1
	mv out_$RESFILE1 $RESFILE1
	rm tmp_$RESFILE1
	cat $RESFILE2 tmp_$RESFILE2 > out_$RESFILE2
	mv out_$RESFILE2 $RESFILE2
	rm tmp_$RESFILE2
done

rm $CONFFILE

exit 0
