#!/bin/bash

for num_nodes in 16 32 64
do 
    mkdir exp_$num_nodes
    sudo nsys profile -w false -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn,osrt --stats=true --capture-range=cudaProfilerApi -o prof_$num_nodes ../../main.py $num_nodes
    sudo nsys stats -f csv --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --output exp_$num_nodes/stats_$num_nodes prof_$num_nodes.sqlite
done

# python get_nsys_plots.py