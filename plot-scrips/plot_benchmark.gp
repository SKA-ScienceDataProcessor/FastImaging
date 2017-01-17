#!/usr/bin/gnuplot
#
# Usage: gnuplot -c plot_benchmark.gp config file1 file2 title 
#

BENCH="bench"
XLABEL="x axis"           # XCOL
FILTERPAR="sup"           # FCOL
XCOL=2    # X axis data file column
YCOL=4    # Y axis data file column
FCOL=3    # Filtered data file column
TCOL=8    # Total data file columns

##########

if(ARGC != 4){
    print ">> ABORTED. Wrong number of input parameters: ", ARGC
    print ">> Usage: gnuplot -c plot_benchmark.gp config file1 file2 title"
    exit
}

file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

CONFIG=ARG1
if(!file_exists(CONFIG)){
    print ">> ABORTED. File '", CONFIG, "' not found!"
    exit
}

load(CONFIG)

FILE1=ARG2
FILE2=ARG3
TITLE=ARG4
print ">> Run gnuplot using:"
print " * Configuration      : ", CONFIG
print " * Reference filename : ", FILE1
print " * Optimized filename : ", FILE2
print " * Append to title    : ", TITLE
BENCH_BS="`echo "@BENCH" | sed 's#_#\\\\\_#g'`"
FILE1_BS="`echo "@FILE1" | sed 's#_#\\\\\_#g'`"
FILE2_BS="`echo "@FILE2" | sed 's#_#\\\\\_#g'`"
TITLE_="`echo "@TITLE" | sed 's# #_#g'`"

if(!file_exists(FILE1)){
    print ">> ABORTED. File '", FILE1, "' not found!"
    exit
}
if(!file_exists(FILE1)){
    print ">> ABORTED. File '", FILE2, "' not found!"
    exit
}

system "sed 's#/# #g' -i ".FILE1
system "sed 's#/# #g' -i ".FILE2

bench_exists(file, bench) = system("cat ".file." | grep -q ^".bench." && echo '1' || echo '0'") + 0
if(!bench_exists(FILE1, BENCH)){
    print ">> ABORTED. Benchmark function not found in ", FILE1
    exit
}
if(!bench_exists(FILE2, BENCH)){
    print ">> ABORTED. Benchmark function not found in ", FILE2
    exit
}

get_colvals(file, bench, fcol) = system("cat ".file." | grep ^".bench." | awk '{print $".fcol."}i' | sort -n | uniq | xargs")

#system "awk '{print $".XCOL."}' ".FILE1." | tr '[a-z]' '[A-Z]' |  numfmt --from=iec --invalid=ignore > tmpfile"
#system "awk -i inplace 'FNR==NR{a[NR]=$1;next}{$".XCOL."=a[FNR]}1' tmpfile ".FILE1
#system "awk '{print $".XCOL."}' ".FILE2." | tr '[a-z]' '[A-Z]' |  numfmt --from=iec --invalid=ignore > tmpfile"
#system "awk -i inplace 'FNR==NR{a[NR]=$1;next}{$".XCOL."=a[FNR]}1' tmpfile ".FILE2
#system "rm tmpfile"

XVALS=get_colvals(FILE1, BENCH, XCOL)
print " * X axis values      : ", XVALS

if(FCOL > 0){
    FILTERVALS=get_colvals(FILE1, BENCH, FCOL)
    print " * Linepoint values   : ", FILTERVALS
}

##########

set term post eps 16 enhanced color
set xlabel XLABEL
set key bottom right
set grid

##########

PLOT="Speedup"
PLOT_="`echo "@PLOT" | sed 's# #_#g'`"
OUTPUT1=BENCH.'-'.TITLE_.'-'.PLOT_.'.eps'
print " * Output plot 1      : ", OUTPUT1

set output OUTPUT1
set title PLOT." - ".BENCH_BS." - ".TITLE font ",18"
set ylabel PLOT
if(exists("FILTERVALS")){
    plot for [VAR in FILTERVALS] '< paste '.FILE1.' '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(column(FCOL)==VAR ? (column(YCOL)/column(YCOL+TCOL)) : 1/0) t FILTERPAR.'='.VAR w linespoints
}else{
    plot '< paste '.FILE1.' '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(column(YCOL)/column(YCOL+TCOL)) t '' w linespoints
}

##########

PLOT="Running time reduction"
PLOT_="`echo "@PLOT" | sed 's# #_#g'`"
OUTPUT2=BENCH.'-'.TITLE_.'-'.PLOT_.'.eps'
print " * Output plot 2      : ", OUTPUT2

set output OUTPUT2 
set title PLOT." - ".BENCH_BS." - ".TITLE font ",18"
set ylabel PLOT.' (%)'
if(exists("FILTERVALS")){
    plot for [VAR in FILTERVALS] '< paste '.FILE1.' '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(column(FCOL)==VAR ? 100*(column(YCOL)-column(YCOL+TCOL))/(column(YCOL)) : 1/0) t FILTERPAR.'='.VAR w linespoints
}else{
    plot '< paste '.FILE1.' '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(100*(column(YCOL)-column(YCOL+TCOL))/(column(YCOL))) t '' w linespoints
}

##########

PLOT="Running times"
PLOT_="`echo "@PLOT" | sed 's# #_#g'`"
OUTPUT3=BENCH.'-'.TITLE_.'-'.PLOT_.'.eps'
print " * Output plot 3      : ", OUTPUT3

set output OUTPUT3
set title PLOT." - ".BENCH_BS." - ".TITLE font ",18"
set ylabel PLOT.' [ns]'
set logscale y 10
if(exists("FILTERVALS")){
    plot for [VAR in FILTERVALS] '< cat '.FILE1.' | grep ^'.BENCH.'' u (column(XCOL)):(column(FCOL)==VAR ? (column(YCOL)) : 1/0) t FILE1_BS.', '.FILTERPAR.'='.VAR w linespoints, \
    for [VAR in FILTERVALS] '< cat '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(column(FCOL)==VAR ? (column(YCOL)): 1/0) t FILE2_BS.', '.FILTERPAR.'='.VAR w linespoints
}else{
    plot '< cat '.FILE1.' | grep ^'.BENCH.'' u (column(XCOL)):(column(YCOL)) t FILE1_BS w linespoints, \
    '< cat '.FILE2.' | grep ^'.BENCH.'' u (column(XCOL)):(column(YCOL)) t FILE2_BS w linespoints
}

##########

print ">> Done"
