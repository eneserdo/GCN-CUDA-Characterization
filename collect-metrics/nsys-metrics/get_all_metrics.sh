#!/bin/bash

for num_nodes in 16 64 256 1024 4096
do 
    # mkdir exp_nn$num_nodes
    # sudo nsys profile -w false -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn --stats=true --capture-range=cudaProfilerApi -o exp_nn$num_nodes/prof_nn$num_nodes ../../main.py -n $num_nodes > /dev/null
    # sudo nsys stats -f csv --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --output exp_nn$num_nodes/stats_nn$num_nodes exp_nn$num_nodes/prof_nn$num_nodes.sqlite > /dev/null

    sudo /opt/nvidia/nsight-compute/2023.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export exp_nn$num_nodes/compute_full_nn$num_nodes --force-overwrite --set full --nvtx main.py -n $num_nodes

    echo exp_nn$num_nodes done!
    sleep 2
done

# mkdir exp_skewed1024
# sudo nsys profile -w false -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn --stats=true --capture-range=cudaProfilerApi -o exp_skewed1024/prof_skewed1024 ../../main.py -n 1024 -s > /dev/null
# sudo nsys stats -f csv --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --output exp_skewed1024/stats_skewed1024 exp_skewed1024/prof_skewed1024.sqlite > /dev/null

sudo /opt/nvidia/nsight-compute/2023.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export exp_skewed1024/compute_full_skewed1024 --force-overwrite --set full --nvtx main.py -n $num_nodes

echo exp_skewed1024 done!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
echo all done!


# python get_nsys_plots.py