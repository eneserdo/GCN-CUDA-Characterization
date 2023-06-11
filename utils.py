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


def get_random_barabasi_albert_graph(N: int, sparsity: float, num_classes: int, num_feat: int):
    avg_degree=int(N * sparsity / 2)
    G = nx.barabasi_albert_graph(N, avg_degree)
    
    A=nx.to_numpy_array(G)
    
    permuated_list=np.random.permutation(N)
    A_permuted=nx.to_numpy_array(G, nodelist=permuated_list)
    A_permuted=torch.from_numpy(A_permuted)
    
    xs, ys = torch.where(A_permuted)
    edge_index=torch.stack([xs, ys])
    x=torch.rand((N, num_feat))
    y=torch.randint(0, num_classes, (1,1))
    data=Data(x=x, edge_index=edge_index, y=y)
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

    degrees=[]
    # for i in range(N):
        # degrees.append(degree(data.edge_index[0], data.num_nodes))
    degrees=degree(data.edge_index[0], data.num_nodes)
    plt.figure()
    
    
    plt.hist(degrees, 64)
    # plt.show()
    plt.savefig(get_unique_path(data_name+str(N) +".png"))


def save_graph_img(data: Data, N, data_name):
    """ Not fully tested yet """

    mat=to_dense_adj(data.edge_index, data.batch)
    plt.figure()
    plt.imshow(mat[0], cmap="gray", vmin=0, vmax=1)
    # plt.show()
    plt.savefig(get_unique_path(data_name+str(N) +".png"))


