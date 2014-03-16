#!/bin/sh

dt=0.05
t_max=10
t=0

cnt=0

rm -rf ./output
mkdir output

while (( `echo "$t < $t_max" | bc` == 1 )); do
    echo "Computing for t = $t"
    n=`printf '%06d' $cnt`
    mpirun -n 1 ./solve 1000 $t 0.0001 > output/output_$n
     # Plot the results
    gnuplot <<-GNUPLOT
    set yrange [0:1.1];
    set term png; set output "./output/plot_${n}.png"; plot "./output/output_$n"
GNUPLOT

    t=`echo $t + $dt | bc`
    cnt=$(( $cnt + 1 ))
done

# Make an animation
convert -delay 10 -loop 1 ./output/plot_* ./output/movie.gif

echo 'Done!'
