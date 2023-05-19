import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum

width = 0.35 
num_nodes_list=[16,32,64]
num_exp=len(num_nodes_list)

data_nvtx=[]
durations=[]
ranges=[]

for num_nodes in num_nodes_list:
    csv_nvtx = pd.read_csv(os.path.join("collect-metrics", "nsys-metrics", f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_nvtx_sum.csv"))
    # csv_kern = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", "stats_nn{num_nodes}_cuda_gpu_kern_sum.csv"))
    # csv_api = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", "stats_nn{num_nodes}_cuda_api_sum.csv"))
    
    # read avg (ns)
    durations.append(np.array(csv_nvtx["Avg (ns)"].to_list()))
    ranges.append(np.array(csv_nvtx["Range"].to_list()))
    # ranges=ranges.sort()
    # sorted(ranges)
    
    
first_range=ranges[0]
for r in ranges:
    if not (first_range == r).all():
        print(first_range)
        print(r)


durations=np.array(durations)


ind = np.arange(num_exp) 

for i in range(num_exp):
    plt.bar(ind+width, durations[i, :], width, color='royalblue')


# add some
# plt.set_ylabel('Scores')
# plt.set_title('Scores by group and gender')
# plt.set_xticks(ind + width / 2)
# ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

# ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

# plt.show()

plt.savefig("asd.png")
