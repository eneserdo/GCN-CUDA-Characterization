import os
import matplotlib.pyplot as plt
import pandas as pd

# nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum


data_nvtx={}

num_nodes_list=[]
for num_nodes in num_nodes_list:
    csv_nvtx = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", "stats_nn{num_nodes}_nvtx_sum.csv"))
    # csv_kern = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", "stats_nn{num_nodes}_cuda_gpu_kern_sum.csv"))
    # csv_api = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", "stats_nn{num_nodes}_cuda_api_sum.csv"))
    
    # read avg (ns)



