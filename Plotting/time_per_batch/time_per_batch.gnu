load "../style/default.gnu"
load "../style/term.gnu"
set out "time_per_batch.tex"

set lmargin 10

set key top right offset -5,0

set title "Mean Inference Time vs Batch Size" 
set xlabel '$\log_{2}(\text{Batch Size})$' 
set ylabel "Mean Inference Time (ms)" 

set xrange[-0.5:10.5]
set logscale y

plot "onnx_time_per_batch.dat" u (log($1)/log(2)):2:3 w yerrorbars ls 1 title "ONNXRunTime",\
    "torch_time_per_batch.dat" u (log($1)/log(2)):2:3 w yerrorbars ls 2 title "Torch" ,\
    "onnx_time_per_batch_4_threads.dat" u (log($1)/log(2)):2:3 w yerrorbars ls 3 title "ORT 4 Threads",\
    "torch_time_per_batch_4_threads.dat" u (log($1)/log(2)):2:3 w yerrorbars ls 4 title "Torch 4 Threads",\
