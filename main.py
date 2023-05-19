#!/home/stannis/.conda/envs/enes/bin/python

# Enes Erdogan
import argparse
import cProfile
import random
import subprocess
import time

import numpy as np
# import nvtx
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import GCN_Model
from utils import get_minutes, get_random_graph, get_random_skewed_graph, save_graph_img

NUM_EPOCH = 1
NUM_BATCH = 16
NUM_NODES_TIMES_BATCH_SIZE = 2**16 # = num_nodes * batch_size. In every iteration this much data is loaded into memory
NUM_CLASSES = 10
NUM_FEAT = 128
# DATASET_LEN = NUM_BATCH*BATCH_SIZE (will be declared later)

"""
    # Effect of graph size:
        Training configs (Number of nodes, Batch size):
        16 - 4096
        256 - 256
        4096 - 16
        18384 - 4
"""


def train_one_epoch(
    model: GCN_Model,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer,
    device,
):
    """ This was written as a separate function just for convenience during cpu profiling. """

    loss=0
    i=0
    for dataBatch in loader:
        optimizer.zero_grad()

        # rng = nvtx.start_range(message="copyToGPU", color="blue")
        dataBatch = dataBatch.to(device)
        # nvtx.end_range(rng)

        # rng = nvtx.start_range(message="model", color="blue")
        pred = model(dataBatch.x, dataBatch.edge_index, dataBatch.batch, i)
        # nvtx.end_range(rng)

        # rng = nvtx.start_range(message="loss", color="blue")
        loss = criterion(pred, dataBatch.y[:, 0])
        # nvtx.end_range(rng)

        # rng = nvtx.start_range(message="backward", color="blue")
        loss.backward()
        # nvtx.end_range(rng)

        # rng = nvtx.start_range(message="update", color="blue")
        optimizer.step()
        # nvtx.end_range(rng)
        i+=1
    
    return loss

def main(args):
    
    batch_size = args.batch_size if args.batch_size != -1 else int(NUM_NODES_TIMES_BATCH_SIZE / args.num_nodes)
    DATASET_LEN = batch_size*NUM_BATCH

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print("Device: ", device)

    # Generate data
    dataset = []
    
    # Select function: skewed or not   
    get_graph = get_random_skewed_graph if args.skewed else get_random_graph

    for i in range(DATASET_LEN):
        data = get_graph(args.num_nodes, args.sparsity, NUM_CLASSES, NUM_FEAT)
        dataset.append(data)
        # if i==10:
        #     save_graph_img(data, args.num_nodes)

    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=args.num_workers)  # worker_init_fn=seed_worker)

    # Set the model
    model = GCN_Model(NUM_FEAT, NUM_CLASSES).to(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Train
    print("Training starts...")
    for epoch in range(NUM_EPOCH):
        start = time.perf_counter()

        loss = train_one_epoch(model, loader, criterion, optimizer, device)

        stop = time.perf_counter()

        print("### Epoch:", epoch, "loss:", round(loss.item(), 4), "time:", get_minutes(stop - start)) # type: ignore
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_nodes", type=int, default=2**4, help="...")
    parser.add_argument("-bs", "--batch_size", type=int, default=-1, help="...")
    parser.add_argument("--num_workers", type=int, default=1, help="...")
    parser.add_argument("--sparsity", type=float, default=0.3, help="...")
    parser.add_argument("-s", "--skewed", action="store_true", help="...")
    parser.add_argument("-cpu", "--use_cpu", action="store_true", help="...")
    parser.add_argument("-p", "--cprofile", action="store_true", help="...")
    parser.add_argument("--seed", type=int, default=34, help="...")

    args = parser.parse_args()
    print(args)
    if args.cprofile:
        # try:
        #     cProfile.run("main(args)", "cProfile-output.prof")
        # except:
        #     print("ERROR!")
        # finally:
        #     subprocess.run("snakeviz cProfile-output.prof", shell=True)
        
        cProfile.run("main(args)", "cProfile-output.prof")
        subprocess.run("snakeviz cProfile-output.prof", shell=True)
        
    else:
        main(args)
