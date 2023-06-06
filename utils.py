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
from torch_geometric.utils import to_dense_adj, degree, add_self_loops
import matplotlib.pyplot as plt
import networkx as nx


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

    x=torch.rand((N, num_feat))
    y=torch.randint(0, num_classes, (1,1))
    
    # order=torch.randperm(N)
    # K=int(sparsity*N*N)
    
    # src=torch.randn((K))*(N**(0.5))
    # src=torch.abs(src)
    # src=torch.round(src).type(torch.long)
    # dst=torch.randint(0, N, (K, 1)).reshape(K)
    
    # edge_index=torch.stack([order[src], dst])

    # data=Data(x=x, edge_index=edge_index, y=y)

    avg_degree=int(2 * sparsity * (N - 1))
    graph = nx.barabasi_albert_graph(N, avg_degree)
    # graph = nx.generators.random_graphs.powerlaw_cluster_graph(N, int(sparsity * N), 0.05)
    
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    edge_index, _ = add_self_loops(edge_index)

    data = Data(x=x, edge_index=edge_index, y=y)

    data.validate()
    return data

def get_minutes(elapsed):
    return f"{round((elapsed)//60)}.{round((elapsed)%60)}"


def get_unique_path(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename +"_" + str(counter) + extension
        counter += 1

    return path

def save_node_degree_dist(data: Data, N, data_name):
    """ Not fully tested yet """

    degrees=[]
    # for i in range(N):
        # degrees.append(degree(data.edge_index[0], data.num_nodes))
    degrees=degree(data.edge_index[0], data.num_nodes)
    plt.figure()
    plt.hist(degrees)
    # plt.show()
    plt.savefig(get_unique_path(data_name+str(N) +".png"))


def save_graph_img(data: Data, N, data_name):
    """ Not fully tested yet """

    mat=to_dense_adj(data.edge_index, data.batch)
    plt.figure()
    plt.imshow(mat[0], cmap="gray", vmin=0, vmax=1)
    # plt.show()
    plt.savefig(get_unique_path(data_name+str(N) +".png"))


