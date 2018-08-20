#!/usr/bin/gnuplot
#
# Usage: gnuplot -c plot_benchmark.gp resfile 
#


XLABEL="Maximum Support Size"
YLABEL="Running time [s]"
TITLE="STP timings varying support size"

set term post eps 20 enhanced color
set xlabel XLABEL
set key top left
set grid

TITLE_="`echo "@TITLE" | sed 's# #_#g'`"
OUTPUT="fig_".TITLE_.".eps"
set output OUTPUT
set title TITLE font ",16"
set ylabel YLABEL
plot \
ARG1 u 1:22 t "Total" w linespoints lw 2.5 ps 1.5, \
ARG1 u 1:5 t "Normalise" w linespoints lw 2.5 ps 1.5, \
ARG1 u 1:4 t "FFT" w linespoints lw 2.5 ps 1.5, \
ARG1 u 1:3 t "Grid loop" w linespoints lw 2.5 ps 1.5, \
ARG1 u 1:2 t "Grid init" w linespoints lw 2.5 ps 1.5

