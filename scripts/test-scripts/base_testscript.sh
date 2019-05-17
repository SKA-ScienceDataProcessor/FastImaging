#!/bin/bash

if [ $# != 4 ]
then
	echo "Missing Field, Parameter-list and ConfigFile and DataFile."
	exit 1
fi

FIELD=$1
CONFFILE_ORIG=$3
MSDATA=$4
RESFILE_BENCH="res_bench_"$3"_"$4"_"$FIELD".txt"
RESFILE1="res_accu_"$3"_"$4"_"$FIELD".txt"

echo_run()
{
	echo "$1"
	eval "$1"
}


echo -n > $RESFILE_BENCH

if [[ $MSDATA == fivesrcdata* ]] ;
then
	echo -n > $RESFILE1
fi

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
	echo_run "./reduce $CONFFILE $MSDATA > /dev/null"
	echo_run "cat logfile.txt | grep ' \[benchmark\] ' | grep ' = ' | awk 'BEGIN {printf \"$i\"}{printf \" %s\", \$NF} END {printf \"\\n\"}' > tmp_$RESFILE_BENCH"
	if [[ $MSDATA == fivesrcdata* ]] ;
	then
		echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'Island 0:' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {printf \"$i\"}{printf \" %s\", \$NF} END {printf \" \"}' > tmp_$RESFILE1"
		echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'Island 1:' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {}{printf \" %s\", \$NF} END {printf \" \"}' >> tmp_$RESFILE1"
		echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'Island 2:' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {}{printf \" %s\", \$NF} END {printf \" \"}' >> tmp_$RESFILE1"
		echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'Island 3:' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {}{printf \" %s\", \$NF} END {printf \" \"}' >> tmp_$RESFILE1"
		echo_run "cat logfile.txt | grep ' \[sources\] ' | grep 'Island 4:' | grep '(?<=extremum_val=).*(?=, extremum_x_idx)' -oP | awk 'BEGIN {}{printf \" %s\", \$NF} END {printf \"\\n\"}' >> tmp_$RESFILE1"
	fi	

        cat $RESFILE_BENCH tmp_$RESFILE_BENCH > out_$RESFILE_BENCH
	mv out_$RESFILE_BENCH $RESFILE_BENCH
	rm tmp_$RESFILE_BENCH
	if [[ $MSDATA == fivesrcdata* ]] ;
	then
		cat $RESFILE1 tmp_$RESFILE1 > out_$RESFILE1
		mv out_$RESFILE1 $RESFILE1
		rm tmp_$RESFILE1
	fi
done

rm $CONFFILE

exit 0
