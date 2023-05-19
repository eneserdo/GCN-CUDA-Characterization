#!/bin/bash

for num_nodes in 16 32 64
do 
    sudo nsys profile -t cuda,nvtx --stats=true -o main_$num_nodes ../main.py $num_nodes
    sudo nsys stats -f table --report nvtx_sum,cuda_gpu_kern_sum --output stats_$num_nodes main_$num_nodes.sqlite
done