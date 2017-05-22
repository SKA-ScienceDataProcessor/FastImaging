# Scripts for plotting benchmark results
## Usage
- Run benchmark tests of function X for two diferent STP implementations
- Copy (or redirect) benchmark output results of function X to plain text files (one file per STP implementation) 
- Run gnuplot script using configuration file for function X, text file results and plot title (word describing difference between the two STP implementations):
$ gnuplot -c plot_benchmark.gp <configuration> <results-first-stp> <results-second-stp> <plot-title> 

## Example
```
$ cd ~/STP_v1/Release/benchmark
$ ./populate_kernel_cache_benchmark > ~/RES_stp_v1.txt
$ cd ~/STP_v2/Release/benchmark
$ ./populate_kernel_cache_benchmark > ~/RES_stp_v2.txt
$ cd ~/plot-scripts
$ gnuplot -c plot_benchmark.gp populate_kernel_cache_benchmark.gp ~/RES_stp_v1.txt ~/RES_stp_v2.txt armafield
```
