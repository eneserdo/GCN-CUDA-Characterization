import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_nvtx_figure():
    # for file_name in ["nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum"]:

    normalize={}
    propagate={}
    linear={}

    file_name="nvtx_sum"

    experiments=[e for e in os.listdir(os.path.join("collect-metrics", "nsys-metrics")) if e.startswith("exp_")]
    num_exp=len(experiments)

    experiments=sorted(experiments)

    for exp in experiments:
        # csv = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_{file_name}.csv"))
        csv = pd.read_csv(os.path.join("collect-metrics", "nsys-metrics", exp, f"stats_{exp[4:]}_{file_name}.csv"))

        # read avg (ns)

        avg=csv["Avg (ns)"].to_list()
        Range=csv["Range"].to_list()
        
        normalize[exp]=avg[Range.index("Normalize")]
        propagate[exp]=avg[Range.index("Propagate")]
        linear[exp]=avg[Range.index("Linear")]
    

    # fcn=col_names[0]
    # for cn in col_names:
    #     assert 


    data_nn={exp:[normalize[exp], propagate[exp], linear[exp]] for exp in experiments if exp.startswith("exp_nn")}
    data_data_irregular={exp:[normalize[exp], propagate[exp], linear[exp]] for exp in experiments if not exp.startswith("exp_nn")}
    


    # for i in range(num_exp):
    #     plt.bar(range(3), normalize[i, :], width, color='royalblue')


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
    for i in range(3):
        r=[x + i*bar_width for x in range(num_exp)]
        plt.bar(r, values[i], width=bar_width, edgecolor='white', label=experiments[i])
        # plt.xticks([r + i*bar_width], [f'Metric {i}'])


    # Add xticks and labels
    # plt.xlabel('Metric')
    plt.ylabel('millisecond')
    plt.xticks([r + bar_width for r in range(len(values[0]))], ['Normalization', 'Linear', 'Propagation'])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    # add some
    # plt.set_ylabel('Scores')
    # plt.set_title('Scores by group and gender')
    # plt.set_xticks(ind + width / 2)
    # ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

    # ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

    # plt.show()

    plt.savefig("asd.png")

def save_kernel_figure():
    pass

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
