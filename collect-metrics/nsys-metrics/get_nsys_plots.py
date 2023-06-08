import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from natsort import natsorted

def save_nvtx_figure():

    normalize={}
    linear={}
    propagate={}

    file_name="nvtx_sum"

    experiments=[e for e in os.listdir(os.path.join("collect-metrics", "nsys-metrics")) if e.startswith("exp_")]
    num_exp=len(experiments)

    experiments=natsorted(experiments)

    for exp in experiments:
        # csv = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_{file_name}.csv"))
        csv = pd.read_csv(os.path.join("collect-metrics", "nsys-metrics", exp, f"stats_{exp[4:]}_{file_name}.csv"))

        avg=csv["Avg (ns)"].to_list()
        Range=csv["Range"].to_list()
        
        normalize[exp]=avg[Range.index("Normalize")]
        linear[exp]=avg[Range.index("Linear")]
        propagate[exp]=avg[Range.index("Propagate")]
    


    data_nn={exp:[normalize[exp], linear[exp], propagate[exp]] for exp in experiments if exp.startswith("exp_nn")}
    data_skewed={exp:[normalize[exp], linear[exp], propagate[exp]] for exp in experiments if not exp.startswith("exp_nn")}
    

    # Extract the experiments and values
    experiments = list(data_nn.keys())
    values = list(data_nn.values())

    # Set the width of the bars
    bar_width = 0.2

    # # Set the positions of the bars on the x-axis
    # r1 = range(len(values[0]))
    # r2 = [x + bar_width for x in r1]
    # r3 = [x + bar_width for x in r2]

    # # Create the bar charts
    # plt.bar(r1, values[0], color='r', width=bar_width, edgecolor='white', label=experiments[0])
    # plt.bar(r2, values[1], color='g', width=bar_width, edgecolor='white', label=experiments[1])
    # plt.bar(r3, values[2], color='b', width=bar_width, edgecolor='white', label=experiments[2])

    # r = 
    for i in range(num_exp):
        r=[x + i*bar_width for x in range(3)]
        plt.bar(r, values[i], width=bar_width, edgecolor='white', label=experiments[i])
        # plt.xticks([r + i*bar_width], [f'Metric {i}'])



    plt.ylabel('nanoseconds')
    plt.xticks([r + bar_width for r in range(len(values[0]))], ['Normalization', 'Linear', 'Propagation'])

    plt.legend()
    #plt.show()

    plt.savefig("nvtx_execution_times.png")

def save_kernel_figure():
    index_select={}
    scatter={}

    file_name="cuda_gpu_kern_sum"

    experiments=[e for e in os.listdir(os.path.join("collect-metrics", "nsys-metrics")) if e.startswith("exp_")]
    num_exp=len(experiments)

    experiments=natsorted(experiments)

    for exp in experiments:
        # csv = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_{file_name}.csv"))
        csv = pd.read_csv(os.path.join("collect-metrics", "nsys-metrics", exp, f"stats_{exp[4:]}_{file_name}.csv"))

        avg=csv["Avg (ns)"].to_list()
        Name=csv["Name"].to_list()
        
        for i, n in enumerate(Name):
            if n.find("scatter")!=-1:
                scatter_index=i

        for i, n in enumerate(Name):
            if n.find("indexSelect")!=-1:
                index_select_index=i


        scatter[exp]=avg[scatter_index]
        index_select[exp]=avg[index_select_index]
    


    data_nn={exp:[scatter[exp], index_select[exp]] for exp in experiments if exp.startswith("exp_nn")}
    data_skewed={exp:[normalize[exp], linear[exp], propagate[exp]] for exp in experiments if not exp.startswith("exp_nn")}
    

    # Extract the experiments and values
    experiments = list(data_nn.keys())
    values = list(data_nn.values())

    # Set the width of the bars
    bar_width = 0.2

    # # Set the positions of the bars on the x-axis
    # r1 = range(len(values[0]))
    # r2 = [x + bar_width for x in r1]
    # r3 = [x + bar_width for x in r2]

    # # Create the bar charts
    # plt.bar(r1, values[0], color='r', width=bar_width, edgecolor='white', label=experiments[0])
    # plt.bar(r2, values[1], color='g', width=bar_width, edgecolor='white', label=experiments[1])
    # plt.bar(r3, values[2], color='b', width=bar_width, edgecolor='white', label=experiments[2])

    # r = 
    plt.figure()
    for i in range(num_exp):
        r=[x + i*bar_width for x in range(2)]
        plt.bar(r, values[i], width=bar_width, edgecolor='white', label=experiments[i])
        # plt.xticks([r + i*bar_width], [f'Metric {i}'])



    plt.ylabel('nanoseconds')
    plt.xticks([r + bar_width for r in range(len(values[0]))], ['scatter', 'indexSelect'])

    plt.legend()
    #plt.show()

    plt.savefig("kernel_execution_times.png")


if __name__ == '__main__':
    save_nvtx_figure()
    save_kernel_figure()



"""
void at::native::elementwise_kernel<
                                        (int)128, 
                                        (int)2, 
                                        void at::native::gpu_kernel_impl<
                                                                        at::native::BinaryFunctor<
                                                                                                    float, 
                                                                                                    float, 
                                                                                                    float, 
                                                                                                    at::native::MulFunctor<float>
                                                                                                >
                                                                        >
                                                                        (at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]
                                        >
                                        (int, T3)
"""
