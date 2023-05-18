import glob
import json
import os
from os.path import join as opj
from typing import List, Tuple
import json
import numpy as np

import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
import torch_geometric.nn as geometric_nn
from torch_geometric.utils import to_dense_adj, degree
import matplotlib.pyplot as plt


def get_random_graph(N: int, sparsity: float, num_classes: int, num_feat: int):
    graph = torch.rand((N, N))
    graph = graph < sparsity
    xs, ys = torch.where(graph)
    edge_index=torch.stack([xs, ys])
    x=torch.rand((N, num_feat))
    y=torch.randint(0, num_classes, (1,1))
    data=Data(x=x, edge_index=edge_index, y=y)
    data.validate()
    return data


def get_random_skewed_graph(N: int, sparsity: float, num_classes: int, num_feat: int):
    """ Not fully tested yet """

    graph = torch.zeros((N, N))
    x=torch.rand((N, num_feat))
    y=torch.randint(0, num_classes, (1,1))
    
    order=torch.randperm(N)
    K=int(sparsity*N*N)
    
    src=torch.randn((K))*(N**(0.5))
    src=torch.abs(src)
    src=torch.round(src).type(torch.long)
    dst=torch.randint(0, N, (K, 1))
    
    edge_index=torch.stack([order[src], dst])

    data=Data(x=x, edge_index=edge_index, y=y)
    data.validate()
    return data

def get_minutes(elapsed):
    return f"{round((elapsed)//60)}.{round((elapsed)%60)}"



def get_node_degree_dist(data: Data, N):
    """ Not fully tested yet """

    degrees=[]
    for i in range(N):
        degrees.append(degree(data.edge_index[i], data.num_nodes))

    plt.plot(degrees)
    plt.plot()


def save_graph_img(data: Data, N):
    """ Not fully tested yet """

    mat=to_dense_adj(data.edge_index, data.batch)
    plt.imshow(mat)
    plt.show()
    get_node_degree_dist(data, N)
