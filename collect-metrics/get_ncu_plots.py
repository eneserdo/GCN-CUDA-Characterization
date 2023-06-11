import ncu_report
import os
from natsort import natsorted
import glob
import_dir=glob.glob("/usr/local/cuda/nsight-compute-*/extras/python")

if len(import_dir) !=1:
    raise ImportError
else:
    import_dir=import_dir[0]
    
import sys
sys.path.append(import_dir)


achieved_occupancy="sm__warps_active.avg.pct_of_peak_sustained_active"
l1_hit_rate="l1tex__t_sector_hit_rate.pct"  	
l2_hit_rate="lts__t_sector_hit_rate.pct"
compute_throughput="sm__throughput.avg.pct_of_peak_sustained_elapsed"
memory_throughput="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

no_eligible="smsp__issue_inst0.avg.pct_of_peak_sustained_active"
eligible="smsp__issue_active.avg.pct_of_peak_sustained_active"


def avr_metric_by_nvtx_range(report, nvtx_range, metric, verbose=False):
    assert report.num_ranges()==1
    my_range = report.range_by_idx(0)
    arr=[]
    prev_i=0
    for i in range(my_range.num_actions()):
        kernel=my_range.action_by_idx(i)
        
        try:
            if kernel.nvtx_state().domain_by_id(0).push_pop_ranges()[0]==nvtx_range:
                val=kernel.metric_by_name(metric).as_double()
                if verbose:
                    print(f"{i-prev_i})", val)
                arr.append(val)
                prev_i=i
        except:
            pass
    
    return sum(arr)/len(arr)

def avr_metric_by_kernel_name(report, kernel_name, metric, verbose=False):
    assert report.num_ranges()==1
    my_range = report.range_by_idx(0)
    arr=[]
    prev_i=0
    for i in range(my_range.num_actions()):
        kernel=my_range.action_by_idx(i)
        
        if kernel.name().find(kernel_name) !=-1:
            val=kernel.metric_by_name(metric).as_double()
            arr.append(val)
            if verbose:
                print(f"{i-prev_i})", kernel.name(), val)    
            prev_i=i

    return sum(arr)/len(arr)  


def main():

    # experiments_names=[e for e in os.listdir(".") if e.startswith("exp_nn")]
    # experiments_names=natsorted(experiments_names)
    experiments_names.append(my_context)
    normalize={}
    linear={}
    propagate={}

    # data_nn={exp:[normalize[exp], linear[exp], propagate[exp]] for exp in experiments}

    metrics_to_collect=[l1_hit_rate, l2_hit_rate, achieved_occupancy, memory_throughput, compute_throughput]

    for m in metrics_to_collect:
        normalize[m]={}
        linear[m]={}
        propagate[m]={}
        for exp in experiments_names:
            exp=ncu_report.load_report(exp)
            normalize[m][exp]=avr_metric_by_nvtx_range(exp, "Normalize", m)
            linear[m][exp]=avr_metric_by_nvtx_range(exp, "Linear", m)
            propagate[m][exp]=avr_metric_by_nvtx_range(exp, "Propagate", m)



if __name__ == '__main__':
    main()