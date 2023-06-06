#!/bin/bash

for num_nodes in 16 64 256 1024 2048
do 
    mkdir exp_nn$num_nodes
    sudo nsys profile -w false -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn --stats=true --capture-range=cudaProfilerApi -o exp_nn$num_nodes/prof_nn$num_nodes ../../main.py -n $num_nodes > /dev/null
    sudo nsys stats -f csv --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --output exp_nn$num_nodes/stats_nn$num_nodes exp_nn$num_nodes/prof_nn$num_nodes.sqlite > /dev/null
    echo exp_nn$num_nodes done!
    sleep 2
done


echo all done!

# python get_nsys_plots.py