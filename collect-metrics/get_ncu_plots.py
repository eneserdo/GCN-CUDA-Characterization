import glob
import_dir=glob.glob("/usr/local/cuda/nsight-compute-*/extras/python")

if len(import_dir) !=1:
    raise ImportError
else:
    import_dir=import_dir[0]
    
import sys
sys.path.append(import_dir)
import ncu_report


def main():
    my_report = ncu_report.load_report("/home/enerd/parallel_cuda/project_data/nn1024_full.ncu-rep")


if __name__ == '__main__':
    main()