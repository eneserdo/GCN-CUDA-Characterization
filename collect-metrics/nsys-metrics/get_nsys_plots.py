import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

width=0.4
num_nodes_list=[16,32,64]
num_exp=len(num_nodes_list)




# for file_name in ["nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum"]:
for file_name in ["nvtx_sum"]:
    durations=[]
    col_names=[]
    data={i:{} for i in num_nodes_list}

    for num_nodes in num_nodes_list:
        # csv = pd.read_csv(os.path.join(f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_{file_name}.csv"))
        csv = pd.read_csv(os.path.join("collect-metrics", "nsys-metrics", f"exp_nn{num_nodes}", f"stats_nn{num_nodes}_{file_name}.csv"))

        # read avg (ns)

        temp_dur=np.array(csv["Avg (ns)"].to_list())
        temp_cn=np.array(csv["Range"].to_list())
        
        order=np.argsort(temp_cn)
        temp_cn=np.sort(temp_cn)
        temp_dur=temp_dur[order]

        durations.append(temp_dur)
        col_names.append(temp_cn)

    durations=np.array(durations)
    
    # fcn=col_names[0]
    # for cn in col_names:
    #     assert 


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



"""void at::native::elementwise_kernel<
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
                                        (int, T3)"""